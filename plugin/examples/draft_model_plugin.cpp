// A small draft model behind the plugin interface (common/speculative-plugin.h): the plugin loads the
// model with libllama, keeps its own context and drafts greedily, as --spec-type draft-simple does.
// It needs nothing from the host but the token history. Sequence s of the draft context holds only
// verified tokens, so nothing there is ever rolled back (that matters for recurrent models such as
// Qwen3.5); each call copies it to sequence n_seq + s and drafts there.
//
// Settings (--spec-plugin-args):
//   model=FNAME   the draft model, with the target's vocabulary (required)
//   ngl=N         layers on the GPU (default: all)
//   ctx=N         context size per sequence (default 8192)
//   p_min=P       stop a draft when the top token's probability is below P (default 0: never)
#include "speculative-plugin.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

struct state {
    llama_model     * model = nullptr;
    llama_context   * ctx   = nullptr;
    llama_batch_ext * batch = nullptr;
    int32_t           n_vocab = 0;
    int32_t           n_batch = 0;
    float             p_min   = 0.0f;

    uint32_t n_seq = 0;
    int32_t  n_in_batch = 0;
    std::vector<std::vector<int32_t>> kv; // per sequence: the verified tokens in the draft context, from position 0
    size_t n_rebuilds = 0;                // times a sequence was decoded again from the start (a new request)

    // without p_min, a greedy sampler on the device per draft sequence, so a draft step reads one token
    // instead of the logits
    std::vector<llama_sampler *> chains;

    ~state() {
        for (size_t i = 0; i < chains.size(); ++i) {
            if (chains[i]) {
                llama_set_sampler(ctx, (llama_seq_id) (n_seq + i), nullptr);
                llama_sampler_free(chains[i]);
            }
        }
        if (batch) {
            llama_batch_ext_free(batch);
        }
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
        if (n_rebuilds > 0) {
            fprintf(stderr, "draft_model plugin: %zu sequences decoded again from the start\n", n_rebuilds);
        }
    }

    int32_t add(int32_t seq_id, int32_t token, llama_pos pos, bool logits) {
        const int32_t idx = llama_batch_ext_add_token(batch, seq_id, token);
        if (idx >= 0) {
            n_in_batch++;
            llama_batch_ext_set_pos(batch, idx, &pos);
            llama_batch_ext_set_output_logits(batch, idx, logits);
        }
        return idx;
    }

    // a verified token, to sequence seq_id
    void commit(int32_t seq_id, int32_t token) {
        add(seq_id, token, (llama_pos) kv[seq_id].size(), false);
        kv[seq_id].push_back(token);
    }

    bool process() {
        if (n_in_batch == 0) {
            return true;
        }
        n_in_batch = 0;
        const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        llama_batch_ext_clear(batch);
        return ret == 0;
    }
};

uint64_t vocab_hash(const llama_vocab * vocab) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int32_t id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
        const char * text = llama_vocab_get_text(vocab, id);
        for (size_t i = 0; i < strlen(text) + 1; ++i) {
            h = (h ^ (uint8_t) text[i]) * 0x100000001b3ULL;
        }
    }
    return h;
}

void * dm_init(const llama_spec_plugin_init_params * p, char * err, size_t err_size) {
    std::string path;
    int32_t ngl = -1, n_ctx = 8192;
    float p_min = 0.0f;
    for (size_t i = 0; i < p->n_args; ++i) {
        const std::string k = p->keys[i], v = p->values[i];
        if (k == "model") {
            path = v;
        } else if (k == "ngl") {
            ngl = std::atoi(v.c_str());
        } else if (k == "ctx") {
            n_ctx = std::atoi(v.c_str());
        } else if (k == "p_min") {
            p_min = std::strtof(v.c_str(), nullptr);
        } else {
            snprintf(err, err_size, "draft_model: unknown setting %s", k.c_str());
            return nullptr;
        }
    }
    if (path.empty()) {
        snprintf(err, err_size, "draft_model: model=FNAME is required");
        return nullptr;
    }

    auto s = std::make_unique<state>();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    s->model = llama_model_load_from_file(path.c_str(), mparams);
    if (!s->model) {
        snprintf(err, err_size, "draft_model: cannot load %s", path.c_str());
        return nullptr;
    }
    const llama_vocab * vocab = llama_model_get_vocab(s->model);
    if (p->vocab_hash != 0 && vocab_hash(vocab) != p->vocab_hash) {
        snprintf(err, err_size, "draft_model: %s has another vocabulary than the target", path.c_str());
        return nullptr;
    }

    auto cparams = llama_context_default_params();
    cparams.n_seq_max = 2 * p->n_seq; // verified tokens in s, drafts in n_seq + s
    cparams.n_ctx     = (uint32_t) n_ctx * 2 * p->n_seq;
    if (p->ctx_tgt) {
        cparams.n_threads       = llama_n_threads(p->ctx_tgt);
        cparams.n_threads_batch = llama_n_threads_batch(p->ctx_tgt);
    }
    s->ctx = llama_init_from_model(s->model, cparams);
    if (!s->ctx) {
        snprintf(err, err_size, "draft_model: cannot create a context for %s", path.c_str());
        return nullptr;
    }
    s->batch   = llama_batch_ext_init(s->ctx);
    s->n_vocab = llama_vocab_n_tokens(vocab);
    s->n_batch = (int32_t) llama_n_batch(s->ctx);
    s->p_min   = p_min;
    s->n_seq   = p->n_seq;
    s->kv.resize(p->n_seq);

    bool sampled_on_device = false;
    if (p_min == 0.0f) {
        sampled_on_device = true;
        s->chains.assign(p->n_seq, nullptr);
        for (uint32_t i = 0; i < p->n_seq; ++i) {
            llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            if (llama_set_sampler(s->ctx, (llama_seq_id) (p->n_seq + i), chain)) {
                s->chains[i] = chain;
            } else {
                llama_sampler_free(chain);
                sampled_on_device = false;
            }
        }
    }
    fprintf(stderr, "draft_model plugin: %s, tokens picked on the %s\n", path.c_str(), sampled_on_device ? "device" : "host");
    return s.release();
}

void dm_free(void * st) {
    delete static_cast<state *>(st);
}

void dm_begin(void *, int32_t, const int32_t *, size_t) {
    // nothing: the next draft call finds how much of the context still matches
}

// argmax of the logits, and its probability when p_min needs it
int32_t pick(const state & s, int32_t idx, float & p_top) {
    p_top = 1.0f;
    if (s.p_min == 0.0f) {
        const llama_token t = llama_get_sampled_token_ith(s.ctx, idx);
        if (t != LLAMA_TOKEN_NULL) {
            return t;
        }
    }
    const float * logits = llama_get_logits_ith(s.ctx, idx);
    int32_t best = 0;
    for (int32_t t = 1; t < s.n_vocab; ++t) {
        if (logits[t] > logits[best]) {
            best = t;
        }
    }
    p_top = 1.0f;
    if (s.p_min > 0.0f) {
        double sum = 0.0;
        for (int32_t t = 0; t < s.n_vocab; ++t) {
            sum += std::exp((double) logits[t] - logits[best]);
        }
        p_top = (float) (1.0 / sum);
    }
    return best;
}

void dm_draft(void * st, llama_spec_plugin_draft * drafts, size_t n_drafts) {
    auto & s = *static_cast<state *>(st);
    auto * mem = llama_get_memory(s.ctx);

    // 1. sequence s: the verified tokens except the last sampled one, which goes with the drafts
    const size_t n_left_max = (size_t) std::max(1, s.n_batch / (int32_t) (n_drafts + 1));
    for (size_t i = 0; i < n_drafts; ++i) {
        auto & r = drafts[i];
        r.n_out = 0;
        auto & kv = s.kv[r.seq_id];
        const size_t n_verified = r.n_committed - 1;
        size_t n_keep = 0;
        while (n_keep < kv.size() && n_keep < n_verified && kv[n_keep] == r.tokens[n_keep]) {
            n_keep++;
        }
        if (n_keep < kv.size()) { // a new request
            if (!llama_memory_seq_rm(mem, r.seq_id, (llama_pos) n_keep, -1)) {
                llama_memory_seq_rm(mem, r.seq_id, -1, -1);
                n_keep = 0;
                s.n_rebuilds++;
            }
            kv.resize(n_keep);
        }
        // a long catch-up (a new prompt) goes in chunks of its own, so the rest of every sequence fits one batch
        while (n_verified - kv.size() > n_left_max) {
            const size_t end = std::min(n_verified, kv.size() + (size_t) s.n_batch);
            while (kv.size() < end) {
                s.commit(r.seq_id, r.tokens[kv.size()]);
            }
            if (!s.process()) {
                return;
            }
        }
    }
    for (size_t i = 0; i < n_drafts; ++i) {
        auto & r = drafts[i];
        while (s.kv[r.seq_id].size() < r.n_committed - 1) {
            s.commit(r.seq_id, r.tokens[s.kv[r.seq_id].size()]);
        }
    }
    if (!s.process()) {
        return;
    }

    // 2. sequence n_seq + s: a copy of s, then the last sampled token and any prefix, then the drafts
    std::vector<int32_t>   i_out(n_drafts, -1);
    std::vector<llama_pos> pos(n_drafts);
    for (size_t i = 0; i < n_drafts; ++i) {
        auto & r = drafts[i];
        const int32_t tmp = (int32_t) s.n_seq + r.seq_id;
        llama_memory_seq_rm(mem, tmp, -1, -1);
        llama_memory_seq_cp(mem, r.seq_id, tmp, -1, -1);
        pos[i] = (llama_pos) s.kv[r.seq_id].size();
        for (size_t k = r.n_committed - 1; k < r.n_tokens; ++k) {
            i_out[i] = s.add(tmp, r.tokens[k], pos[i]++, k + 1 == r.n_tokens);
        }
    }
    if (!s.process()) {
        return;
    }
    for (;;) {
        bool any = false;
        for (size_t i = 0; i < n_drafts; ++i) {
            auto & r = drafts[i];
            if (i_out[i] < 0) {
                continue;
            }
            float p_top;
            const int32_t tok = pick(s, i_out[i], p_top);
            i_out[i] = -1;
            if (p_top < s.p_min) {
                continue;
            }
            r.out[r.n_out++] = tok;
            if (r.n_out < r.n_out_max) {
                i_out[i] = s.add((int32_t) s.n_seq + r.seq_id, tok, pos[i]++, true);
                any = true;
            }
        }
        if (!any || !s.process()) {
            break;
        }
    }
}

void dm_accept(void *, int32_t, int32_t, int32_t) {
    // nothing: the next draft call sees which tokens were accepted
}

const llama_spec_plugin api = {
    /* .api_version = */ LLAMA_SPEC_PLUGIN_API_VERSION,
    /* .init        = */ dm_init,
    /* .free        = */ dm_free,
    /* .begin       = */ dm_begin,
    /* .draft       = */ dm_draft,
    /* .accept      = */ dm_accept,
};

} // namespace

PLUGIN_EXPORT const llama_spec_plugin * llama_spec_plugin_get(void) {
    return &api;
}
