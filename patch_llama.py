#!/usr/bin/env python3
"""Add the MemDraft drafter to a llama.cpp source tree as `--spec-type ngram-table`.

Written against llama.cpp 7d7f028. Every edit names the exact text it replaces and stops if that
text is missing or ambiguous, so on another commit it fails loudly instead of patching the wrong
place. memdraft.h is copied in unchanged.

    python patch_llama.py /path/to/llama.cpp

New server options:
    --spec-table-file FNAME     the table from build_table.py (or export_reranker.py)
    --spec-table-reranker FNAME pick among the proposals with this reranker (export_reranker.py)
    --spec-table-tau P          draft while the chain's chance of full acceptance is at least P
    --spec-table-n-max N        longest draft
    --spec-table-source S       both | table | context
    --spec-table-pad            always draft exactly n-max tokens, so the batch shape never changes
    --spec-table-const N        benchmarking: always draft N copies of the last token (no table)
    --spec-table-const-period P benchmarking: with --spec-table-const, draft only at every P-th step
    --spec-table-no-rs          do not ask for recurrent-state snapshots (the checkpoint path)
    --spec-table-head-first     the draft model (e.g. draft-mtp) drafts the first token, the table the rest;
                                needs both in --spec-type
    --spec-accept-ratio R       lossy: when sampling greedily, also accept a drafted token whose probability
                                is at least R times the model's top token's (default 1.0, lossless; any drafter)
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

IMPL = r'''
// MemDraft: lookups only, no draft model. See memdraft.h.
struct common_speculative_impl_ngram_table : public common_speculative_impl {
    common_params_speculative_ngram_table params;

    memdraft::table tab; // read-only, shared by all sequences

    memdraft::reranker rr; // read-only, shared by all sequences; used when --spec-table-reranker is given
    memdraft::ple_reranker ple_rr; // experimental scorer sharing retained GGUF rows

    std::vector<memdraft::drafter> drafters;

    std::vector<size_t> n_steps; // per sequence, for --spec-table-const-period

    llama_tokens history;

    common_speculative_impl_ngram_table(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_TABLE, n_seq,
                std::max(params.ngram_table.n_max, params.ngram_table.n_const))
        , params(params.ngram_table) {
        SPC_TRC("%s", "adding speculative implementation 'ngram-table'\n");

        if (this->params.n_const > 0) {
            SPC_WRN("ngram-table drafts %d copies of the last token at every %d. step - for benchmarking only\n",
                    this->params.n_const, this->params.n_period);
        } else {
            std::string err;
            if (!this->params.reranker.empty() && !this->params.ple_reranker.empty()) {
                GGML_ABORT("choose --spec-table-reranker or --spec-table-ple-reranker");
            }
            if (!tab.load(this->params.file, err)) {
                SPC_ERR("%s\n", err.c_str());
                GGML_ABORT("Couldn't read the ngram-table file (--spec-table-file)");
            }
            SPC_INF("ngram-table: %" PRIu64 " entries, contexts up to %u tokens, %.1f MiB, tau = %.3f, n_max = %d, table = %d, context = %d, pad = %d\n",
                    tab.n_entries, tab.K, (double) tab.slots.size() * sizeof(uint64_t) / 1024 / 1024,
                    (double) this->params.tau, this->params.n_max, this->params.table, this->params.context, this->params.pad);
            if (!this->params.ple_reranker.empty()) {
                if (!ple_rr.load(this->params.ple_reranker, err) || tab.version != 2 || ple_rr.K != tab.K) {
                    SPC_ERR("PLE reranker/table mismatch or load failure: %s\n", err.c_str());
                    GGML_ABORT("Couldn't load --spec-table-ple-reranker");
                }
            }
            if (!this->params.reranker.empty()) {
                if (!rr.load(this->params.reranker, err)) {
                    SPC_ERR("%s\n", err.c_str());
                    GGML_ABORT("Couldn't read the ngram-table reranker (--spec-table-reranker)");
                }
                SPC_INF("ngram-table: reranker %u-%u-%u-%u, %.1f MiB\n", rr.n_x + rr.d * (1 + rr.n_ctx), rr.h1, rr.h2, rr.h3,
                        (double) rr.bytes() / 1024 / 1024);
            }
        }

        n_steps.assign(n_seq, 0);
        drafters.resize(n_seq);
        for (auto & d : drafters) {
            d.pad         = this->params.pad;
            d.tab         = &tab;
            d.use_static  = this->params.table;
            d.use_context = this->params.context;
            d.tau         = this->params.tau;
            d.n_max       = this->params.n_max;
            if (!this->params.ple_reranker.empty() && this->params.n_const == 0) {
                d.picker = memdraft::ple_reranker::callback;
                d.picker_data = &ple_rr;
            }
            if (!this->params.reranker.empty() && this->params.n_const == 0) {
                const std::string why = d.check(rr);
                if (!why.empty()) {
                    SPC_ERR("%s\n", why.c_str());
                    GGML_ABORT("ngram-table: this reranker does not fit this table");
                }
                d.rr = &rr;
            }
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        drafters[seq_id].begin();
    }

    bool process(const common_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            if (params.n_const > 0) {
                if (n_steps[seq_id]++ % params.n_period == 0) {
                    result.assign(params.n_const, dp.id_last);
                }
                continue;
            }

            // everything committed so far: the prompt, then the token sampled last
            history.assign(dp.prompt->begin(), dp.prompt->end());
            history.push_back(dp.id_last);

            drafters[seq_id].draft(history.data(), history.size(), result);
        }
    }

    // --spec-table-head-first: `head` is the draft model's first token (or none); the table drafts the rest
    // from the committed tokens plus that guess, which stays out of the request index.
    void draft_after(common_speculative_draft_params & dp, llama_seq_id seq_id, const llama_tokens & head) {
        auto & result = *dp.result;
        if (head.empty()) {
            history.assign(dp.prompt->begin(), dp.prompt->end());
            history.push_back(dp.id_last);
            drafters[seq_id].draft(history.data(), history.size(), result);
            return;
        }
        history.assign(dp.prompt->begin(), dp.prompt->end());
        history.push_back(dp.id_last);
        const size_t n_commit = history.size();
        history.insert(history.end(), head.begin(), head.end());

        tail.clear();
        drafters[seq_id].draft(history.data(), history.size(), tail, n_commit);

        result.assign(head.begin(), head.end());
        for (size_t i = 0; i < tail.size() && (int) result.size() < params.n_max; ++i) {
            result.push_back(tail[i]);
        }
    }

    llama_tokens tail;

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};
'''

ARGS = r'''    add_opt(common_arg(
        {"--spec-table-ple-reranker"}, "FNAME",
        "ngram-table: experimental scorer sharing GGUF ngram rows (ple_experiment.py .mdp)",
        [](common_params & params, const std::string & value) {
            params.speculative.ngram_table.ple_reranker = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-file"}, "FNAME",
        "ngram-table: the table built by build_table.py",
        [](common_params & params, const std::string & value) {
            params.speculative.ngram_table.file = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-reranker"}, "FNAME",
        "ngram-table: pick among the proposals with this reranker (export_reranker.py; needs a version 2 table)",
        [](common_params & params, const std::string & value) {
            params.speculative.ngram_table.reranker = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-tau"}, "P",
        string_format("ngram-table: draft while the chain's chance of full acceptance is at least P (default: %.2f)", (double) params.speculative.ngram_table.tau),
        [](common_params & params, const std::string & value) {
            params.speculative.ngram_table.tau = std::stof(value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-n-max"}, "N",
        string_format("ngram-table: longest draft (default: %d)", params.speculative.ngram_table.n_max),
        [](common_params & params, int value) {
            if (value < 1 || value > 64) {
                throw std::invalid_argument("ngram-table n-max must be between 1 and 64 inclusive");
            }
            params.speculative.ngram_table.n_max = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-source"}, "{both,table,context}",
        "ngram-table: draft from the table, from the current request, or from both (default: both)",
        [](common_params & params, const std::string & value) {
            if (value != "both" && value != "table" && value != "context") {
                throw std::invalid_argument("ngram-table source must be both, table or context");
            }
            params.speculative.ngram_table.table   = value != "context";
            params.speculative.ngram_table.context = value != "table";
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-const"}, "N",
        "ngram-table: benchmarking only, always draft N copies of the last token (default: 0 = off)",
        [](common_params & params, int value) {
            if (value < 0 || value > 64) {
                throw std::invalid_argument("ngram-table const must be between 0 and 64 inclusive");
            }
            params.speculative.ngram_table.n_const = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-pad"},
        "ngram-table: always draft exactly n-max tokens (best guesses past tau), so the batch shape never changes",
        [](common_params & params) {
            params.speculative.ngram_table.pad = true;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-const-period"}, "P",
        "ngram-table: benchmarking only, with --spec-table-const draft at every P-th step (default: 1)",
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram-table const period must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_table.n_period = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-head-first"},
        "ngram-table: the draft model of --spec-type (e.g. draft-mtp) drafts the first token "
        "and the table continues from it, up to --spec-table-n-max tokens in all; needs both in --spec-type",
        [](common_params & params) {
            params.speculative.ngram_table.head_first = true;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-table-no-rs"},
        "ngram-table: do not ask for recurrent-state snapshots; a rejected draft then restores a checkpoint and decodes again",
        [](common_params & params) {
            params.speculative.ngram_table.rs_rollback = false;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-accept-ratio"}, "R",
        string_format("also accept a drafted token whose probability is at least R times the top token's, when sampling "
                      "greedily (default: %.2f = lossless)", (double) params.sampling.spec_accept_ratio),
        [](common_params & params, const std::string & value) {
            const float r = std::stof(value);
            if (!(r > 0.0f && r <= 1.0f)) {
                throw std::invalid_argument("spec-accept-ratio must be in (0, 1]");
            }
            params.sampling.spec_accept_ratio = r;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
'''

# --spec-table-head-first: the table's turn first asks the draft model for one token
DISPATCH_OLD = r'''    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

'''
DISPATCH_NEW = r'''    // --spec-table-head-first: the draft model drafts only as the table's first token
    common_speculative_impl_ngram_table * table = nullptr;
    common_speculative_impl *             head  = nullptr;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_NGRAM_TABLE) {
            auto * t = static_cast<common_speculative_impl_ngram_table *>(impl.get());
            table = t->params.head_first ? t : nullptr;
        } else if (impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) {
            head = head ? head : impl.get();
        }
    }
    if (!table || !head) {
        table = nullptr, head = nullptr;
    }

    for (auto & impl : spec->impls) {
        if (impl.get() == head) {
            continue;   // it drafted inside the table's turn
        }
        if (impl.get() == table) {
            {
                common_time_meas tm(head->t_draft_us, !head->gen_perf);
                head->draft(dparams);
                head->n_call_draft++;
            }
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
                auto & dp = dparams[seq_id];
                if (!dp.drafting) {
                    continue;
                }
                llama_tokens first;
                first.swap(*dp.result);
                if (first.size() > 1) {
                    first.resize(1);
                }
                if (!first.empty()) {
                    head->n_gen_drafts++;
                    head->n_gen_tokens++;
                }
                dp.result_q = nullptr;   // the table's tokens carry no candidates: verify by sample-and-match
                table->draft_after(dp, seq_id, first);
            }
            impl->n_call_draft++;
        } else {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

'''

# --spec-table-head-first: the draft model is asked for one token only
HEAD_N_MAX = [
    (r'''            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
''', r'''            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(head_params(config.params), n_seq));
'''),
    (r'''            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
''', r'''            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(head_params(config.params), n_seq));
'''),
    (r'''    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
''', r'''    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    // --spec-table-head-first uses only the draft model's first token, so it drafts just that one
    const bool head_first = params.ngram_table.head_first &&
        std::any_of(configs.begin(), configs.end(), [](const common_speculative_config & c) {
            return c.type == COMMON_SPECULATIVE_TYPE_NGRAM_TABLE; });
    auto head_params = [&](const common_params_speculative & p) {
        common_params_speculative q = p;
        if (head_first) {
            q.draft.n_max = 1;
            q.draft.n_min = 0;
        }
        return q;
    };

    for (const common_speculative_config & config : configs) {
'''),
]

EDITS = {
    'common/common.h': [
        ('''    float   temp               = 0.80f;  // <= 0.0 to sample greedily, 0.0 to not output probabilities
''', '''    float   temp               = 0.80f;  // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   spec_accept_ratio  = 1.00f;  // speculative, greedy: also accept a drafted token with at least this share of the top token's probability (1.0 = lossless)
'''),
        ('''    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
''', '''    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_NGRAM_TABLE,   // MemDraft: a prebuilt table plus short in-request lookups
'''),
        ('''struct common_params_speculative_ngram_cache {
''', '''struct common_params_speculative_ngram_table {
    std::string file;               // table built by build_table.py or export_reranker.py
    std::string reranker;           // optional reranker from export_reranker.py
    std::string ple_reranker;       // experimental shared-GGUF scorer from ple_experiment.py

    float   tau         = 0.3f;     // draft while the chain's chance of full acceptance is at least this
    int32_t n_max       = 4;        // longest draft
    bool    head_first  = false;    // the draft model (draft-mtp) drafts the first token, the table the rest
    bool    table       = true;     // draft from the table
    bool    context     = true;     // draft from the current request
    bool    pad         = false;    // always draft exactly n_max tokens, so the batch shape never changes
    int32_t n_const     = 0;        // benchmarking: always draft this many copies of the last token
    int32_t n_period    = 1;        // benchmarking: with n_const, draft only at every n_period-th step
    bool    rs_rollback = true;     // ask for recurrent-state snapshots so a rejected draft needs no checkpoint
};

struct common_params_speculative_ngram_cache {
'''),
        ('''    common_params_speculative_ngram_cache ngram_cache;

    bool has_dft() const {''', '''    common_params_speculative_ngram_cache ngram_cache;
    common_params_speculative_ngram_table ngram_table;

    bool has_dft() const {'''),
        ('''        return needs_rs_seq ? draft.n_max : 0u;
''', '''        // ngram-table drafts are short, so per-token snapshots are cheaper than a checkpoint and a second decode
        const bool table_rs = ngram_table.rs_rollback &&
            std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_NGRAM_TABLE) != types.end();

        const uint32_t n_draft = needs_rs_seq ? uint32_t(draft.n_max) : 0u;
        const uint32_t n_table = table_rs ? uint32_t(std::max(ngram_table.n_max, ngram_table.n_const)) : 0u;

        return std::max(n_draft, n_table);
'''),
    ],
    'common/speculative.cpp': [
        ('''#include "ngram-mod.h"
''', '''#include "ngram-mod.h"
#include "memdraft.h"
#include "ple_reranker.h"
'''),
        ('''    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
''', '''    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"ngram-table",   COMMON_SPECULATIVE_TYPE_NGRAM_TABLE}
'''),
        ('''
struct common_speculative {
''', IMPL + '''
struct common_speculative {
'''),
        ('''        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
''', '''        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        case COMMON_SPECULATIVE_TYPE_NGRAM_TABLE:   return "ngram-table";
'''),
        ('''            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
''', '''            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_TABLE:
                n_max = std::max(n_max, std::max(spec->ngram_table.n_max, spec->ngram_table.n_const));
                break;
'''),
        (DISPATCH_OLD, DISPATCH_NEW),
        *HEAD_N_MAX,
        ('''        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
''', '''        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 12);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
'''),
        ('''        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
''', '''        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        // after the drafters that fire only on a long exact repeat and then draft dozens of tokens;
        // the table drafts at nearly every step, so whatever is listed after it rarely gets a turn
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_TABLE);
'''),
        ('''            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                impls.push_back(create_state_ngram_cache(config, n_seq));
                break;
            }
''', '''            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                impls.push_back(create_state_ngram_cache(config, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_TABLE: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_table>(config.params, n_seq));
                break;
            }
'''),
    ],
    'common/sampling.cpp': [
        ('''    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);
''', '''    size_t i = 0;
    for (; i < draft.size(); i++) {
        llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        // MemDraft --spec-accept-ratio: a drafted token the model rates at least `ratio` times its own
        // top choice is kept (lossy below 1.0). p(draft) / p(top) = exp(logit(draft) - logit(top)).
        const float ratio = gsmpl->params.spec_accept_ratio;
        if (id != draft[i] && ratio < 1.0f && gsmpl->grmr == nullptr && !llama_vocab_is_eog(vocab, draft[i])) {
            const float * logits = llama_get_logits_ith(ctx, idxs[i]);
            if (logits[draft[i]] - logits[id] >= std::log(ratio)) {
                id = draft[i];
            }
        }

        common_sampler_accept(gsmpl, id, true);
'''),
    ],
    'common/arg.cpp': [
        ('''    add_opt(common_arg(
        {"--spec-ngram-mod-n-match"}, "N",
''', ARGS + '''    add_opt(common_arg(
        {"--spec-ngram-mod-n-match"}, "N",
'''),
    ],
}


def main():
    tree = Path(sys.argv[1])
    prepared = []
    for name, edits in EDITS.items():
        path = tree / name
        text = path.read_text()
        if 'NGRAM_TABLE' in text or 'spec-table-file' in text or 'spec_accept_ratio' in text:
            raise SystemExit(f'{path} is already patched')
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f'{path}: expected exactly one place to edit, found {text.count(old)}:\n{old}')
            text = text.replace(old, new)
        prepared.append((path, text))
    # Validate the whole revision before touching any source file.
    for path, text in prepared:
        path.write_text(text)
        print(f'patched {path}')
    shutil.copy(HERE / 'memdraft.h', tree / 'common' / 'memdraft.h')
    shutil.copy(HERE / 'ple_reranker.h', tree / 'common' / 'ple_reranker.h')
    print(f"copied memdraft.h to {tree / 'common'}")


if __name__ == '__main__':
    main()
