// A multi-token-prediction head behind the plugin interface (common/speculative-plugin.h): the same
// drafting as --spec-type draft-mtp for a model with one MTP layer (Qwen3.5, Qwen3.5-MoE), written only
// against llama.h. It uses the parts of the interface a token-only drafter does not need: the target
// context (to read its hidden states), process() after every target decode, and per-sequence state.
//
// Settings (--spec-plugin-args):
//   model=FNAME   the MTP head as its own GGUF (convert_hf_to_gguf.py --mtp) (required)
//   ngl=N         layers on the GPU (default: all)
//
// Images in the prompt are skipped: the head sees a gap there, so drafts right after an image may be worse.
#include "speculative-plugin.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

struct state {
    llama_context   * ctx_tgt = nullptr;
    llama_model     * model   = nullptr;
    llama_context   * ctx     = nullptr; // the head's context, paired with the target's
    llama_batch_ext * batch   = nullptr;
    int32_t n_seq   = 0;
    int32_t n_embd  = 0;
    int32_t n_vocab = 0;

    // the target's hidden row for the last committed token of each sequence: the head's input for the next
    // token, and the start of the next draft
    std::vector<std::vector<float>> pending_h;

    // rows of the last target decode, per sequence, so accept() can pick the row of the last accepted token
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_rows;
    std::vector<int32_t> verify_row0;

    // the target's hidden rows stay on the device and the head reads them there (as draft-mtp does)
    bool    dev_rows    = false;
    int32_t n_batch_tgt = 0;

    // Per sequence, keep only the top candidates on the device and take their greedy maximum on the host.
    std::vector<llama_sampler *> chains;

    ~state() {
        for (size_t sq = 0; sq < chains.size(); ++sq) {
            if (chains[sq]) {
                llama_set_sampler(ctx, (llama_seq_id) sq, nullptr);
                llama_sampler_free(chains[sq]);
            }
        }
        if (dev_rows) {
            llama_set_ctx_state_src(ctx, nullptr);
            llama_set_embeddings_nextn_dev(ctx_tgt, false);
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
    }

    int32_t add(int32_t seq_id, int32_t token, llama_pos pos, bool logits) {
        const int32_t idx = llama_batch_ext_add_token(batch, seq_id, token);
        if (idx >= 0) {
            llama_batch_ext_set_pos(batch, idx, &pos);
            llama_batch_ext_set_output_logits(batch, idx, logits);
        }
        return idx;
    }

    void set_h(int32_t idx, const float * h) {
        llama_batch_ext_set_embd_state(batch, idx, { h, 1, (size_t) n_embd });
    }
};

void * mtp_init(const llama_spec_plugin_init_params * p, char * err, size_t err_size) {
    if (!p->ctx_tgt) {
        snprintf(err, err_size, "mtp: needs the target context");
        return nullptr;
    }
    std::string path;
    int32_t ngl = -1;
    for (size_t i = 0; i < p->n_args; ++i) {
        const std::string k = p->keys[i], v = p->values[i];
        if (k == "model") {
            path = v;
        } else if (k == "ngl") {
            ngl = std::atoi(v.c_str());
        } else {
            snprintf(err, err_size, "mtp: unknown setting %s", k.c_str());
            return nullptr;
        }
    }
    if (path.empty()) {
        snprintf(err, err_size, "mtp: model=FNAME is required");
        return nullptr;
    }

    auto s = std::make_unique<state>();
    s->ctx_tgt = p->ctx_tgt;
    s->n_seq   = (int32_t) p->n_seq;

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    mparams.load_mtp     = true;
    s->model = llama_model_load_from_file(path.c_str(), mparams);
    if (!s->model) {
        snprintf(err, err_size, "mtp: cannot load %s", path.c_str());
        return nullptr;
    }
    s->n_embd  = llama_model_n_embd_out(s->model);
    s->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(s->model));
    if (llama_model_n_layer_nextn(s->model) > 1 || s->n_embd != llama_model_n_embd_out(llama_get_model(s->ctx_tgt))) {
        snprintf(err, err_size, "mtp: %s is not a one-layer MTP head for this target", path.c_str());
        return nullptr;
    }

    // as the host makes the draft-mtp context: sized like the target's, one output per sequence
    auto cparams = llama_context_default_params();
    cparams.ctx_type              = LLAMA_CONTEXT_TYPE_MTP;
    cparams.ctx_other             = s->ctx_tgt;
    cparams.n_ctx                 = llama_n_ctx(s->ctx_tgt);
    cparams.n_batch               = llama_n_batch(s->ctx_tgt);
    cparams.n_ubatch              = llama_n_ubatch(s->ctx_tgt);
    cparams.n_seq_max             = llama_n_seq_max(s->ctx_tgt);
    cparams.n_rs_seq              = 0;
    cparams.n_outputs_max         = p->n_seq;
    cparams.n_outputs_max_per_seq = 1;
    cparams.n_threads             = llama_n_threads(s->ctx_tgt);
    cparams.n_threads_batch       = llama_n_threads_batch(s->ctx_tgt);
    s->ctx = llama_init_from_model(s->model, cparams);
    if (!s->ctx) {
        snprintf(err, err_size, "mtp: cannot create the head's context");
        return nullptr;
    }
    s->batch = llama_batch_ext_init(s->ctx);

    llama_set_embeddings_nextn(s->ctx_tgt, true, /*masked*/ false);
    llama_set_embeddings_nextn(s->ctx, true, /*masked*/ true);

    s->pending_h.assign(p->n_seq, std::vector<float>(s->n_embd, 0.0f));
    s->verify_h.assign(p->n_seq, {});
    s->verify_rows.assign(p->n_seq, 0);
    s->verify_row0.assign(p->n_seq, -1);
    s->n_batch_tgt = (int32_t) llama_n_batch(s->ctx_tgt);

    s->dev_rows = llama_set_embeddings_nextn_dev(s->ctx_tgt, true) && llama_set_ctx_state_src(s->ctx, s->ctx_tgt);
    if (!s->dev_rows) {
        llama_set_ctx_state_src(s->ctx, nullptr);
        llama_set_embeddings_nextn_dev(s->ctx_tgt, false);
    }
    bool sampled_on_device = true;
    s->chains.assign(p->n_seq, nullptr);
    for (int32_t sq = 0; sq < s->n_seq; ++sq) {
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        // Match draft-mtp's backend sampler. Device argmax also exports the full vocabulary logits,
        // whereas top-k returns only the shortlisted logits and their token IDs.
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));
        if (llama_set_sampler(s->ctx, sq, chain)) {
            s->chains[sq] = chain;
        } else {
            llama_sampler_free(chain);
            sampled_on_device = false;
        }
    }
    fprintf(stderr, "mtp plugin: %s, hidden rows on the %s, %s\n", path.c_str(),
            s->dev_rows ? "device" : "host", sampled_on_device ? "device top-k" : "host argmax");
    return s.release();
}

void mtp_free(void * st) {
    delete static_cast<state *>(st);
}

void mtp_begin(void *, int32_t, const int32_t *, size_t) {
}

// The head's input at position p is the token at p and the target's hidden row at p - 1: pair each token of
// the target's batch with the row before it, and the first one of each sequence with pending_h.
int32_t mtp_process(void * st, const llama_spec_plugin_batch * b) {
    auto & s = *static_cast<state *>(st);
    std::vector<int32_t> beg(s.n_seq, -1), end(s.n_seq, -1);
    for (int32_t k = 0; k < b->n_tokens; ++k) {
        const int32_t sq = b->seq_id[k];
        end[sq] = k;
        if (beg[sq] < 0) {
            beg[sq] = k;
        }
    }

    // drop what the head holds from the batch's first position on: last step's drafts and rejected tokens
    auto * mem = llama_get_memory(s.ctx);
    for (int32_t sq = 0; sq < s.n_seq; ++sq) {
        if (beg[sq] >= 0) {
            llama_memory_seq_rm(mem, sq, b->pos[beg[sq]], -1);
        }
    }

    if (s.dev_rows) {
        if (b->n_tokens > s.n_batch_tgt) {
            return 1;
        }
        for (int32_t sq = 0; sq < s.n_seq; ++sq) {
            if (beg[sq] >= 0) {
                llama_set_embeddings_nextn_rows(s.ctx_tgt, s.n_batch_tgt + sq, 1, s.pending_h[sq].data());
            }
        }
    }
    const float * h_tgt = s.dev_rows ? nullptr : llama_get_embeddings_nextn(s.ctx_tgt);

    llama_batch_ext_clear(s.batch);
    int32_t n_added = 0;
    for (int32_t k = 0; k < b->n_tokens; ++k) {
        if (b->tokens[k] < 0) {
            continue; // an image entry: the plugin interface gives no embedding for it
        }
        const int32_t sq  = b->seq_id[k];
        const int32_t idx = s.add(sq, b->tokens[k], b->pos[k], false);
        n_added++;
        if (s.dev_rows) {
            llama_batch_ext_set_embd_state_row(s.batch, idx, k == beg[sq] ? s.n_batch_tgt + sq : k - 1);
        } else {
            s.set_h(idx, k == beg[sq] ? s.pending_h[sq].data() : h_tgt + (size_t) (k - 1) * s.n_embd);
        }
    }
    if (n_added > 0 && llama_process(s.ctx, LLAMA_PROCESS_TYPE_DECODE, s.batch) != 0) {
        return 1;
    }

    // keep this batch's rows for accept(); the last one is pending until accept() says otherwise
    for (int32_t sq = 0; sq < s.n_seq; ++sq) {
        if (end[sq] < 0) {
            continue;
        }
        const int32_t n_rows = end[sq] - beg[sq] + 1;
        s.verify_rows[sq] = n_rows;
        if (s.dev_rows) {
            s.verify_row0[sq] = beg[sq];
            llama_get_embeddings_nextn_rows(s.ctx_tgt, beg[sq] + n_rows - 1, 1, s.pending_h[sq].data());
            continue;
        }
        s.verify_h[sq].resize((size_t) n_rows * s.n_embd);
        for (int32_t i = 0; i < n_rows; ++i) {
            std::memcpy(s.verify_h[sq].data() + (size_t) i * s.n_embd, llama_get_embeddings_nextn_ith(s.ctx_tgt, beg[sq] + i),
                        (size_t) s.n_embd * sizeof(float));
        }
        std::memcpy(s.pending_h[sq].data(), s.verify_h[sq].data() + (size_t) (n_rows - 1) * s.n_embd, (size_t) s.n_embd * sizeof(float));
    }
    return 0;
}

int32_t argmax(const state & s, int32_t idx) {
    const uint32_t count = llama_get_sampled_candidates_count_ith(s.ctx, idx);
    const float * logits = llama_get_sampled_logits_ith(s.ctx, idx);
    if (count > 0) {
        const llama_token * candidates = llama_get_sampled_candidates_ith(s.ctx, idx);
        // Backend top-k can return the candidates in any order.
        uint32_t best = 0;
        for (uint32_t i = 1; i < count; ++i) {
            if (logits[i] > logits[best] || (logits[i] == logits[best] && candidates[i] < candidates[best])) {
                best = i;
            }
        }
        return candidates[best];
    }
    if (!logits) {
        logits = llama_get_logits_ith(s.ctx, idx);
    }
    return (int32_t) (std::max_element(logits, logits + s.n_vocab) - logits);
}

void mtp_draft(void * st, llama_spec_plugin_draft * drafts, size_t n_drafts) {
    auto & s = *static_cast<state *>(st);
    auto * mem = llama_get_memory(s.ctx);
    std::vector<int32_t> i_last(n_drafts, -1);

    llama_batch_ext_clear(s.batch);
    for (size_t i = 0; i < n_drafts; ++i) {
        auto & r = drafts[i];
        r.n_out = 0;
        if (r.n_tokens != r.n_committed) {
            continue; // the head drafts from verified tokens only
        }
        // the head still holds the tokens the target rejected after r.pos (the host rolls back only its own
        // contexts): drop them, or the draft attends to them
        llama_memory_seq_rm(mem, r.seq_id, r.pos, -1);
        i_last[i] = s.add(r.seq_id, r.tokens[r.n_committed - 1], r.pos, true);
        s.set_h(i_last[i], s.pending_h[r.seq_id].data());
    }
    // greedy: the head's next input is the token it drafted and its own hidden row
    for (int32_t step = 0;; ++step) {
        if (llama_process(s.ctx, LLAMA_PROCESS_TYPE_DECODE, s.batch) != 0) {
            return;
        }
        std::vector<int32_t> tok(n_drafts, -1);
        std::vector<const float *> h(n_drafts, nullptr);
        for (size_t i = 0; i < n_drafts; ++i) {
            if (i_last[i] >= 0) {
                tok[i] = argmax(s, i_last[i]);
                h[i]   = llama_get_embeddings_nextn_ith(s.ctx, i_last[i]);
            }
        }
        llama_batch_ext_clear(s.batch);
        bool any = false;
        for (size_t i = 0; i < n_drafts; ++i) {
            auto & r = drafts[i];
            if (i_last[i] < 0) {
                continue;
            }
            r.out[r.n_out++] = tok[i];
            i_last[i] = -1;
            if (r.n_out < r.n_out_max) {
                i_last[i] = s.add(r.seq_id, tok[i], r.pos + step + 1, true);
                s.set_h(i_last[i], h[i]);
                any = true;
            }
        }
        if (!any) {
            return;
        }
    }
}

void mtp_accept(void * st, int32_t seq_id, int32_t n_accepted, int32_t) {
    auto & s = *static_cast<state *>(st);
    const int32_t n_rows = s.verify_rows[seq_id];
    if (n_rows <= 0) {
        return;
    }
    const int32_t i_h = std::min(n_accepted, n_rows - 1);
    if (s.dev_rows) {
        if (i_h != n_rows - 1) {
            llama_get_embeddings_nextn_rows(s.ctx_tgt, s.verify_row0[seq_id] + i_h, 1, s.pending_h[seq_id].data());
        }
        return;
    }
    std::memcpy(s.pending_h[seq_id].data(), s.verify_h[seq_id].data() + (size_t) i_h * s.n_embd, (size_t) s.n_embd * sizeof(float));
}

size_t mtp_get_state(void * st, int32_t seq_id, uint8_t * dst, size_t size) {
    auto & s = *static_cast<state *>(st);
    const size_t n = (size_t) s.n_embd * sizeof(float);
    if (dst && size >= n) {
        std::memcpy(dst, s.pending_h[seq_id].data(), n);
    }
    return n;
}

void mtp_set_state(void * st, int32_t seq_id, const uint8_t * src, size_t size) {
    auto & s = *static_cast<state *>(st);
    if (size == (size_t) s.n_embd * sizeof(float)) {
        std::memcpy(s.pending_h[seq_id].data(), src, size);
    }
}

const llama_spec_plugin api = {
    /* .api_version = */ LLAMA_SPEC_PLUGIN_API_VERSION,
    /* .init        = */ mtp_init,
    /* .free        = */ mtp_free,
    /* .begin       = */ mtp_begin,
    /* .draft       = */ mtp_draft,
    /* .accept      = */ mtp_accept,
    /* .process     = */ mtp_process,
    /* .get_state   = */ mtp_get_state,
    /* .set_state   = */ mtp_set_state,
};

} // namespace

PLUGIN_EXPORT const llama_spec_plugin * llama_spec_plugin_get(void) {
    return &api;
}
