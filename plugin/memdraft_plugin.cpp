// MemDraft as a llama.cpp drafter plugin (common/speculative-plugin.h):
//
//   llama-server ... --spec-type plugin --spec-plugin libmemdraft.so --spec-plugin-n-max 3
//       --spec-plugin-args table=T.mdt,reranker=T.mdr,pad=1
//
// Settings (--spec-plugin-args):
//   table=FNAME       the table from build_table.py or export_reranker.py (required)
//   reranker=FNAME    pick among the proposals with this reranker (export_reranker.py)
//   tau=P             draft while the chain's chance of full acceptance is at least P (default 0.3)
//   n_max=N           longest table draft (default: what the host allows after the prefix)
//   pad=0|1           always draft n_max tokens, so the batch shape never changes (default 0)
//   source=S          both | table | context (default both)
//   vocab_hash=HEX    refuse to run unless the target's vocabulary hashes to this (see vocab_hash.py)
//
// With --spec-plugin-prefix the draft model's tokens arrive after the committed ones: the table drafts
// from them but keeps them out of the request index, since they are not verified yet.
#include "memdraft.h"
#include "speculative-plugin.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define MEMDRAFT_EXPORT extern "C" __declspec(dllexport)
#else
#    define MEMDRAFT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct state {
    memdraft::table    tab;
    memdraft::reranker rr;
    bool               has_rr = false;
    int32_t            n_max  = 0;

    std::vector<memdraft::drafter> drafters; // one per sequence
    std::vector<int32_t>           out;
};

void set_err(char * err, size_t size, const std::string & msg) {
    if (err && size > 0) {
        std::snprintf(err, size, "%s", msg.c_str());
    }
}

void * md_init(const llama_spec_plugin_init_params * p, char * err, size_t err_size) {
    std::string table_path, reranker_path, source = "both";
    float       tau        = 0.3f;
    int32_t     n_max      = p->n_max - p->n_prefix;
    bool        pad        = false;
    uint64_t    vocab_hash = 0;

    for (size_t i = 0; i < p->n_args; ++i) {
        const std::string k = p->keys[i], v = p->values[i];
        if (k == "table") {
            table_path = v;
        } else if (k == "reranker") {
            reranker_path = v;
        } else if (k == "tau") {
            tau = std::strtof(v.c_str(), nullptr);
        } else if (k == "n_max") {
            n_max = std::atoi(v.c_str());
        } else if (k == "pad") {
            pad = v == "1" || v == "true";
        } else if (k == "source") {
            source = v;
        } else if (k == "vocab_hash") {
            vocab_hash = std::strtoull(v.c_str(), nullptr, 16);
        } else {
            set_err(err, err_size, "memdraft: unknown setting " + k);
            return nullptr;
        }
    }
    if (table_path.empty()) {
        set_err(err, err_size, "memdraft: table=FNAME is required");
        return nullptr;
    }
    if (source != "both" && source != "table" && source != "context") {
        set_err(err, err_size, "memdraft: source must be both, table or context");
        return nullptr;
    }
    if (n_max < 1) {
        set_err(err, err_size, "memdraft: n_max must be at least 1");
        return nullptr;
    }
    if (vocab_hash != 0 && p->vocab_hash != 0 && vocab_hash != p->vocab_hash) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "memdraft: the table is for vocabulary %016" PRIx64 ", the model has %016" PRIx64,
                      vocab_hash, p->vocab_hash);
        set_err(err, err_size, buf);
        return nullptr;
    }

    auto s = std::make_unique<state>();
    std::string why;
    if (!s->tab.load(table_path, why)) {
        set_err(err, err_size, "memdraft: " + why);
        return nullptr;
    }
    if (!reranker_path.empty()) {
        if (!s->rr.load(reranker_path, why)) {
            set_err(err, err_size, "memdraft: " + why);
            return nullptr;
        }
        s->has_rr = true;
    }
    s->n_max = n_max;
    s->drafters.resize(p->n_seq);
    for (auto & d : s->drafters) {
        d.tab         = &s->tab;
        d.use_static  = source != "context";
        d.use_context = source != "table";
        d.tau         = tau;
        d.n_max       = n_max;
        d.pad         = pad;
        if (s->has_rr) {
            why = d.check(s->rr);
            if (!why.empty()) {
                set_err(err, err_size, "memdraft: " + why);
                return nullptr;
            }
            d.rr = &s->rr;
        }
    }
    std::fprintf(stderr, "memdraft: %" PRIu64 " table entries, contexts up to %u tokens, reranker %s, n_max %d, pad %d, tau %.2f, vocab %016" PRIx64 "\n",
                 s->tab.n_entries, s->tab.K, s->has_rr ? "yes" : "no", n_max, (int) pad, (double) tau, p->vocab_hash);
    return s.release();
}

void md_free(void * st) {
    delete static_cast<state *>(st);
}

void md_begin(void * st, int32_t seq_id, const int32_t * /*prompt*/, size_t /*n_prompt*/) {
    auto * s = static_cast<state *>(st);
    if (seq_id >= 0 && (size_t) seq_id < s->drafters.size()) {
        s->drafters[seq_id].begin();
    }
}

void md_draft(void * st, llama_spec_plugin_draft * drafts, size_t n_drafts) {
    auto * s = static_cast<state *>(st);
    for (size_t i = 0; i < n_drafts; ++i) {
        auto & r = drafts[i];
        r.n_out = 0;
        if (r.seq_id < 0 || (size_t) r.seq_id >= s->drafters.size() || r.n_tokens == 0 || r.n_out_max <= 0) {
            continue;
        }
        auto & d = s->drafters[r.seq_id];
        d.n_max = std::min(s->n_max, r.n_out_max);
        s->out.clear();
        d.draft(r.tokens, r.n_tokens, s->out, r.n_committed);
        r.n_out = std::min<int32_t>((int32_t) s->out.size(), r.n_out_max);
        std::copy(s->out.begin(), s->out.begin() + r.n_out, r.out);
    }
}

void md_accept(void * /*st*/, int32_t /*seq_id*/, int32_t /*n_accepted*/, int32_t /*is_own*/) {
}

const llama_spec_plugin api = {
    /* .api_version = */ LLAMA_SPEC_PLUGIN_API_VERSION,
    /* .init        = */ md_init,
    /* .free        = */ md_free,
    /* .begin       = */ md_begin,
    /* .draft       = */ md_draft,
    /* .accept      = */ md_accept,
};

} // namespace

MEMDRAFT_EXPORT const llama_spec_plugin * llama_spec_plugin_get(void) {
    return &api;
}
