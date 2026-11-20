// Replays token streams (improve/make_live_stream.py) through a built-in n-gram drafter and through the same drafter behind the plugin
// interface (builtin_ngram_plugin), and checks that every draft is identical.
//
//   test_plugin_equiv PLUGIN.so STREAM [STREAM...]
//
// Two sequences run side by side, so the host's batched draft call is exercised too. The target is
// simulated: a stream holds each request's prompt and the answer the model gave, and a draft is accepted
// up to its first token that differs from the answer.
#include "common.h"
#include "speculative.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct request {
    llama_tokens prompt, answer;
};

static std::vector<request> read_stream(const char * path) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    uint32_t n, pad;
    f.read(magic, 8);
    f.read((char *) &n, 4);
    f.read((char *) &pad, 4);
    if (!f || std::memcmp(magic, "MDSTREAM", 8) != 0) {
        fprintf(stderr, "%s: not a stream\n", path);
        exit(1);
    }
    std::vector<request> reqs(n);
    for (auto & r : reqs) {
        uint32_t np, na;
        f.read((char *) &np, 4);
        f.read((char *) &na, 4);
        r.prompt.resize(np);
        r.answer.resize(na);
        f.read((char *) r.prompt.data(), 4 * np);
        f.read((char *) r.answer.data(), 4 * na);
        f.seekg(4 * (std::streamoff) na, std::ios::cur); // a second copy of the answer (the replay's targets)
    }
    return reqs;
}

struct seq_state {
    int          req = -1;  // index of the request in this sequence
    size_t       i   = 0;   // answer tokens committed so far
    llama_tokens history;   // prompt + committed answer tokens, without the last sampled one
    llama_tokens draft_a, draft_b;
};

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s PLUGIN.so STREAM [STREAM...]\n", argv[0]);
        return 1;
    }
    std::vector<request> reqs;
    for (int k = 2; k < argc; ++k) {
        auto r = read_stream(argv[k]);
        reqs.insert(reqs.end(), r.begin(), r.end());
    }

    const char * types[] = { "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache" };
    int failed = 0;
    for (const char * type : types) {
        common_params_speculative pa;
        pa.types = { common_speculative_type_from_name(type) };
        common_speculative_ptr a(common_speculative_init(pa, 2));
        const int32_t n_max = common_speculative_n_max(a.get());

        common_params_speculative pb;
        pb.types = { COMMON_SPECULATIVE_TYPE_PLUGIN };
        pb.plugin.path  = argv[1];
        pb.plugin.n_max = n_max;
        pb.plugin.args  = { { "type", type } };
        common_speculative_ptr b(common_speculative_init(pb, 2));

        seq_state seqs[2];
        size_t next = 0, n_steps = 0, n_drafts = 0, n_drafted = 0, n_accepted = 0, n_diff = 0;
        auto start = [&](llama_seq_id s) {
            auto & st = seqs[s];
            st.req = next < reqs.size() ? (int) next++ : -1;
            if (st.req < 0) {
                return;
            }
            st.i = 0;
            st.history = reqs[st.req].prompt;
            common_speculative_begin(a.get(), s, st.history);
            common_speculative_begin(b.get(), s, st.history);
        };
        start(0);
        start(1);
        while (seqs[0].req >= 0 || seqs[1].req >= 0) {
            for (auto * spec : { a.get(), b.get() }) {
                for (llama_seq_id s = 0; s < 2; ++s) {
                    auto & st = seqs[s];
                    auto & dp = common_speculative_get_draft_params(spec, s);
                    auto & out = spec == a.get() ? st.draft_a : st.draft_b;
                    out.clear();
                    dp = {};
                    if (st.req < 0) {
                        continue;
                    }
                    dp.drafting = true;
                    dp.n_max    = n_max;
                    dp.pos0     = (llama_pos) st.history.size();
                    dp.id_last  = reqs[st.req].answer[st.i];
                    dp.prompt   = &st.history;
                    dp.result   = &out;
                }
                common_speculative_draft(spec);
            }
            for (llama_seq_id s = 0; s < 2; ++s) {
                auto & st = seqs[s];
                if (st.req < 0) {
                    continue;
                }
                const auto & ans = reqs[st.req].answer;
                if (st.draft_a != st.draft_b) {
                    if (n_diff++ < 3) {
                        fprintf(stderr, "%s: request %d, token %zu: built-in drafted %zu tokens, plugin %zu\n",
                                type, st.req, st.i, st.draft_a.size(), st.draft_b.size());
                    }
                }
                size_t n_acc = 0;
                while (n_acc < st.draft_a.size() && st.i + 1 + n_acc < ans.size() && st.draft_a[n_acc] == ans[st.i + 1 + n_acc]) {
                    n_acc++;
                }
                n_steps++;
                n_drafts += !st.draft_a.empty();
                n_drafted += st.draft_a.size();
                n_accepted += n_acc;
                common_speculative_accept(a.get(), s, (uint16_t) n_acc);
                common_speculative_accept(b.get(), s, (uint16_t) n_acc);

                st.history.insert(st.history.end(), ans.begin() + st.i, ans.begin() + st.i + 1 + n_acc);
                st.i += 1 + n_acc;
                if (st.i >= ans.size()) {
                    start(s);
                }
            }
        }
        printf("%-14s n_max %2d  steps %7zu  drafts %6zu  drafted %7zu  accepted %6zu  differing steps %zu\n",
               type, n_max, n_steps, n_drafts, n_drafted, n_accepted, n_diff);
        failed += n_diff > 0;
    }
    printf(failed ? "FAIL: %d drafter(s) differ\n" : "OK: every draft identical\n", failed);
    return failed ? 1 : 0;
}
