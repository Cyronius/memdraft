// llama.cpp's own n-gram drafters behind the plugin interface (common/speculative-plugin.h).
//
// This checks the interface, not a new drafter: the plugin runs one built-in drafter through the public
// speculative.h API, fed only with what the plugin interface hands it. If the interface carries enough,
// the plugin drafts exactly what the built-in drafter drafts (test_plugin_equiv.cpp compares them).
//
// Settings (--spec-plugin-args):
//   type=ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache   (required)
//   size_n, size_m, min_hits      ngram-simple, ngram-map-k, ngram-map-k4v
//   n_match, n_min, n_max         ngram-mod
//   static, dynamic               ngram-cache lookup files
#include "speculative-plugin.h"

#include "common.h"
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#define PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

struct state {
    common_params_speculative params;
    common_speculative_ptr    spec;

    std::vector<llama_tokens> prompts; // per sequence: the committed tokens before the last sampled one
    std::vector<llama_tokens> results;
};

void * bridge_init(const llama_spec_plugin_init_params * p, char * err, size_t err_size) {
    auto s = std::make_unique<state>();
    auto & sp = s->params;
    std::string type;
    for (size_t i = 0; i < p->n_args; ++i) {
        const std::string k = p->keys[i], v = p->values[i];
        const int n = std::atoi(v.c_str());
        if (k == "type") {
            type = v;
        } else if (k == "size_n") {
            sp.ngram_simple.size_n = sp.ngram_map_k.size_n = sp.ngram_map_k4v.size_n = n;
        } else if (k == "size_m") {
            sp.ngram_simple.size_m = sp.ngram_map_k.size_m = sp.ngram_map_k4v.size_m = n;
        } else if (k == "min_hits") {
            sp.ngram_simple.min_hits = sp.ngram_map_k.min_hits = sp.ngram_map_k4v.min_hits = n;
        } else if (k == "n_match") {
            sp.ngram_mod.n_match = n;
        } else if (k == "n_min") {
            sp.ngram_mod.n_min = n;
        } else if (k == "n_max") {
            sp.ngram_mod.n_max = n;
        } else if (k == "static") {
            sp.ngram_cache.lookup_cache_static = v;
        } else if (k == "dynamic") {
            sp.ngram_cache.lookup_cache_dynamic = v;
        } else {
            std::snprintf(err, err_size, "unknown setting %s", k.c_str());
            return nullptr;
        }
    }
    const auto t = common_speculative_type_from_name(type);
    if (type.rfind("ngram-", 0) != 0 || t == COMMON_SPECULATIVE_TYPE_COUNT) {
        std::snprintf(err, err_size, "type must be one of the n-gram drafters, got '%s'", type.c_str());
        return nullptr;
    }
    sp.types = { t };
    try {
        s->spec.reset(common_speculative_init(sp, p->n_seq));
    } catch (const std::exception & e) {
        std::snprintf(err, err_size, "%s", e.what());
        return nullptr;
    }
    if (!s->spec) {
        std::snprintf(err, err_size, "no drafter for type '%s'", type.c_str());
        return nullptr;
    }
    s->prompts.resize(p->n_seq);
    s->results.resize(p->n_seq);
    return s.release();
}

void bridge_free(void * st) {
    delete static_cast<state *>(st);
}

void bridge_begin(void * st, int32_t seq_id, const int32_t * prompt, size_t n_prompt) {
    auto * s = static_cast<state *>(st);
    common_speculative_begin(s->spec.get(), seq_id, llama_tokens(prompt, prompt + n_prompt));
}

void bridge_draft(void * st, llama_spec_plugin_draft * drafts, size_t n_drafts) {
    auto * s = static_cast<state *>(st);
    for (size_t i = 0; i < n_drafts; ++i) {
        const auto & r = drafts[i];
        auto & prompt = s->prompts[r.seq_id];
        prompt.assign(r.tokens, r.tokens + r.n_committed - 1);
        s->results[r.seq_id].clear();

        auto & dp = common_speculative_get_draft_params(s->spec.get(), r.seq_id);
        dp = {};
        dp.drafting = true;
        dp.n_max    = r.n_out_max;
        dp.pos0     = (llama_pos) r.n_committed - 1;
        dp.id_last  = r.tokens[r.n_committed - 1];
        dp.prompt   = &prompt;
        dp.result   = &s->results[r.seq_id];
    }
    common_speculative_draft(s->spec.get()); // all sequences in one call, as the host does
    for (size_t i = 0; i < n_drafts; ++i) {
        auto & r = drafts[i];
        const auto & res = s->results[r.seq_id];
        r.n_out = std::min<int32_t>((int32_t) res.size(), r.n_out_max);
        std::copy(res.begin(), res.begin() + r.n_out, r.out);
    }
}

void bridge_accept(void * st, int32_t seq_id, int32_t n_accepted, int32_t is_own) {
    auto * s = static_cast<state *>(st);
    if (is_own) {
        common_speculative_accept(s->spec.get(), seq_id, (uint16_t) n_accepted);
    }
}

const llama_spec_plugin api = {
    /* .api_version = */ LLAMA_SPEC_PLUGIN_API_VERSION,
    /* .init        = */ bridge_init,
    /* .free        = */ bridge_free,
    /* .begin       = */ bridge_begin,
    /* .draft       = */ bridge_draft,
    /* .accept      = */ bridge_accept,
};

} // namespace

PLUGIN_EXPORT const llama_spec_plugin * llama_spec_plugin_get(void) {
    return &api;
}
