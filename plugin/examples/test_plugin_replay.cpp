// Replays token streams (improve/make_live_stream.py) through one plugin via the host's plugin drafter and
// prints how much of each draft the recorded answers would accept. A check that a plugin drafts sensibly.
//
//   test_plugin_replay PLUGIN.so N_MAX "KEY=VALUE,..." N_REQUESTS STREAM [STREAM...]
#include "common.h"
#include "speculative.h"

#include <cstdio>
#include <cstdlib>
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

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s PLUGIN.so N_MAX KEY=VALUE,... N_REQUESTS STREAM [STREAM...]\n", argv[0]);
        return 1;
    }
    std::vector<request> reqs;
    for (int k = 5; k < argc; ++k) {
        auto r = read_stream(argv[k]);
        reqs.insert(reqs.end(), r.begin(), r.end());
    }
    reqs.resize(std::min(reqs.size(), (size_t) atoi(argv[4])));

    common_params_speculative p;
    p.types = { COMMON_SPECULATIVE_TYPE_PLUGIN };
    p.plugin.path  = argv[1];
    p.plugin.n_max = atoi(argv[2]);
    for (const auto & kv : string_split<std::string>(argv[3], ',')) {
        const auto pos = kv.find('=');
        p.plugin.args.emplace_back(kv.substr(0, pos), kv.substr(pos + 1));
    }
    common_speculative_ptr spec(common_speculative_init(p, 1));

    size_t n_steps = 0, n_drafted = 0, n_accepted = 0, n_first = 0;
    llama_tokens draft;
    for (const auto & r : reqs) {
        llama_tokens history = r.prompt;
        common_speculative_begin(spec.get(), 0, history);
        for (size_t i = 0; i < r.answer.size();) {
            auto & dp = common_speculative_get_draft_params(spec.get(), 0);
            draft.clear();
            dp = {};
            dp.drafting = true;
            dp.n_max    = p.plugin.n_max;
            dp.pos0     = (llama_pos) history.size();
            dp.id_last  = r.answer[i];
            dp.prompt   = &history;
            dp.result   = &draft;
            common_speculative_draft(spec.get());
            size_t n_acc = 0;
            while (n_acc < draft.size() && i + 1 + n_acc < r.answer.size() && draft[n_acc] == r.answer[i + 1 + n_acc]) {
                n_acc++;
            }
            n_steps++;
            n_drafted += draft.size();
            n_accepted += n_acc;
            n_first += n_acc > 0;
            common_speculative_accept(spec.get(), 0, (uint16_t) n_acc);
            history.insert(history.end(), r.answer.begin() + i, r.answer.begin() + i + 1 + n_acc);
            i += 1 + n_acc;
        }
    }
    printf("requests %zu  steps %zu  drafted %zu  accepted %zu (%.1f%%)  first token right %.1f%%  tokens per step %.2f\n",
           reqs.size(), n_steps, n_drafted, n_accepted, 100.0 * n_accepted / std::max<size_t>(1, n_drafted),
           100.0 * n_first / std::max<size_t>(1, n_steps), (double) (n_steps + n_accepted) / std::max<size_t>(1, n_steps));
    return 0;
}
