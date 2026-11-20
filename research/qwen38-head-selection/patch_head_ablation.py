#!/usr/bin/env python3
"""Add runtime PLE-head masking to the pinned research llama.cpp build.

QWEN4EXP_PLE_KEEP_HEADS=0,8 keeps only those heads. Unset keeps all heads.
QWEN4EXP_PLE_SWEEP runs semicolon-separated masks with one model load.
This is for calibration: ship a surgically pruned GGUF without this patch.
"""
import argparse
from pathlib import Path

OLD = '''    cb(emb, "ple_embd", -1);
'''
NEW = '''    // Research-only ablation: no GGUF copies needed for a head sweep.
    if (const char * selected = std::getenv("QWEN4EXP_PLE_KEEP_HEADS")) {
        std::set<int64_t> keep;
        std::stringstream input(selected);
        std::string item;
        while (std::getline(input, item, ',')) {
            size_t used = 0;
            int64_t h = std::stoll(item, &used);
            if (used != item.size() || h < 0 || h >= n_heads) {
                throw std::runtime_error("invalid QWEN4EXP_PLE_KEEP_HEADS");
            }
            keep.insert(h);
        }
        ggml_tensor * masked = nullptr;
        for (int64_t h = 0; h < n_heads; ++h) {
            ggml_tensor * part = ggml_view_2d(ctx0, emb, hparams.ple_head_dim, n_tokens,
                    emb->nb[1], h * hparams.ple_head_dim * emb->nb[0]);
            if (!keep.count(h)) part = ggml_scale(ctx0, part, 0.0f);
            masked = masked ? ggml_concat(ctx0, masked, part, 0) : part;
        }
        emb = masked;
    }
    cb(emb, "ple_embd", -1);
'''

SWEEP_INIT = '''    const char * sweep = std::getenv("QWEN4EXP_PLE_SWEEP");
    if (sweep) {
        if (params.ppl_stride || params.hellaswag || params.winogrande ||
                params.multiple_choice || params.kl_divergence || !params.logits_file.empty()) {
            LOG_ERR("PLE sweep requires ordinary perplexity evaluation\\n");
            return 1;
        }
        // A graph built for one mask cannot be reused for another mask.
        setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    }

'''
SWEEP_EVAL = '''    if (sweep) {
        std::stringstream masks(sweep);
        std::string mask;
        const char * yield_path = std::getenv("ARBITER_YIELD_FILE");
        while (std::getline(masks, mask, ';')) {
            if (yield_path && std::ifstream(yield_path).good()) {
                return 75;
            }
            setenv("QWEN4EXP_PLE_KEEP_HEADS", mask.c_str(), 1);
            std::printf("\\nPLE_SWEEP_START %s\\n", mask.c_str());
            std::fflush(stdout);
            results = perplexity(ctx, params, n_ctx);
            if (!std::isfinite(results.ppl_value) || results.ppl_value <= 0) {
                return 1;
            }
            std::printf("\\nPLE_SWEEP_RESULT %s %.10f\\n", mask.c_str(), results.ppl_value);
            std::fflush(stdout);
        }
    } else '''


def patch_perplexity(path):
    text = path.read_text()
    init = '    // load the model and apply lora adapter, if any\n'
    evaluate = '    if (params.hellaswag) {'
    if 'QWEN4EXP_PLE_SWEEP' in text or text.count(init) != 1 or text.count(evaluate) != 1:
        raise SystemExit('already patched or unexpected perplexity source revision')
    text = text.replace(init, SWEEP_INIT + init).replace(evaluate, SWEEP_EVAL + evaluate.lstrip())
    path.write_text('#include <cstdlib>\n' + text)
    print(f'patched {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--sweep-only', action='store_true',
                        help='add sweep support to a Linux tree with head masking already installed')
    args = parser.parse_args()
    path = args.source / 'src/models/qwen4exp.cpp'
    text = path.read_text()
    already_masked = 'QWEN4EXP_PLE_KEEP_HEADS' in text
    if args.sweep_only:
        if not already_masked:
            parser.error('--sweep-only requires existing Qwen head masking')
    else:
        if already_masked or text.count(OLD) != 1:
            parser.error('already patched or unexpected qwen4exp source revision')
        path.write_text('#include <cstdlib>\n#include <set>\n#include <sstream>\n' + text.replace(OLD, NEW))
        print(f'patched {path}')
    patch_perplexity(args.source / 'tools/perplexity/perplexity.cpp')


if __name__ == '__main__':
    main()
