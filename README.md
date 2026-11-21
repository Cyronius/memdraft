# MemDraft

MemDraft is a speculative-decoding drafter for llama.cpp that runs no model. It guesses the next tokens by looking them up, and the target model checks the guesses as in any speculative decoding. A draft step costs microseconds, so even guesses that are right only some of the time make decoding faster.

It ships as a plugin, `libmemdraft.so`, loaded by llama.cpp's `--spec-type plugin`. That option comes from a small patch to llama.cpp in this repo.

## Results

These numbers are for Qwen3.5-9B Q4_K_M on an AMD Radeon 8060S through Vulkan, with greedy decoding, the table and reranker built from the 9B's own code, and no other process on the GPU (2026-11-18). "Speed" is tokens per second against plain decoding on the same prompts.

| Setup | Coding prompts (12 OSS-Instruct prompts, 2 rounds) | HumanEval (12 problems) |
|---|---:|---:|
| Plain decoding | 38.0 tok/s | 38.2 tok/s |
| **MemDraft, behind `ngram-simple`** | **1.85×** | **3.29×** |
| **MemDraft, behind `ngram-simple` and the model's MTP head** | **2.14×** | **3.57×** |
| llama.cpp `draft-mtp` (the MTP head alone, 3 tokens) | 1.94× | 2.30× |
| llama.cpp `ngram-mod` | 1.08× | 1.78×¹ |
| Qwen3.5-2B as a draft model, 3 tokens | 1.17× | 1.36×² |

¹ From a separate run on the same machine. ² Only 2 problems completed.

- With both MemDraft setups, every prompt in every round ran faster than plain decoding.
- HumanEval answers restate code from the prompt, which suits lookup drafting. The coding prompts are the more typical case.
- **Verification is lossless, but llama.cpp's batched decoding is not bit-exact.** The output text matched plain decoding on 11 of 12 HumanEval problems and on 8 of 24 coding-prompt runs. Where it differs, it splits at a near-tie between two tokens; plain decoding also changes this way when only the batch size changes.
- **Answer quality held up on HumanEval.** Over all 164 problems, pass@1 was 83.5% against 84.8% for plain decoding: two problems flipped. That run used the same table and reranker without `ngram-simple`, in the earlier patched build of llama.cpp.
- **The setup is narrow so far.** It has been measured on one model, on code, on one GPU. On a mix of chat, math and editing prompts, an earlier table built from the 9B's chat and math text gave 1.24× overall, but only 1.09× on chat.
- Raw results are in [`bench/results/`](bench/results/). How MemDraft got here, including what failed, is in [docs/history.md](docs/history.md).

## How it works

At each step MemDraft proposes tokens from two lookups:

1. **A table built ahead of time** from text the target model wrote. For each of the last 1 to 6 tokens of context, it stores the token the model most often wrote next and how often that guess was right on held-out text.
2. **The current request.** The same kind of lookup over the prompt and the tokens generated so far.

A small reranker (an MLP over the proposals and the target's token embeddings) picks among the proposals. MemDraft always drafts the same number of tokens (`pad=1`), so llama.cpp's compute graph keeps its shape and is never rebuilt. A draft call takes about 60 µs with the reranker and 3 µs without it.

It works best in a chain of drafters. llama.cpp's `ngram-simple` drafts first when the last 12 tokens appeared earlier in the request, so long copies come from there. Optionally, the model's own MTP head drafts the first token. MemDraft fills in everywhere else.

## Quick start

You need:

- llama.cpp at commit `edec462`
- a table (`.mdt`) and a reranker (`.mdr`) for your model's tokenizer

The files used above were built for the Qwen3.5/3.6/3.8 tokenizer; they are not published yet. See [Building a table](#building-a-table).

```bash
# 1. Patch and build llama.cpp
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout edec462
git apply /path/to/memdraft/patches/llama-plugin-hook.patch
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j      # or the backend you use

# 2. Build the plugin
/path/to/memdraft/plugin/build.sh /path/to/llama.cpp          # writes plugin/libmemdraft.so

# 3. Serve: ngram-simple first, then MemDraft drafting 3 tokens
llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 \
  --spec-type ngram-simple,plugin --spec-plugin-n-max 3 \
  --spec-plugin /path/to/memdraft/plugin/libmemdraft.so \
  --spec-plugin-args table=TABLE.mdt,reranker=TABLE.mdr,pad=1,vocab_hash=c52114e74bc3a20f
```

To put the MTP head in front as well, add it as a draft model and let it draft the first token:

```bash
  --spec-type ngram-simple,draft-mtp,plugin --spec-draft-model HEAD.gguf --spec-draft-p-min 0 \
  --spec-plugin-prefix 1 --spec-plugin-n-max 4
```

To make the head, export it from the Hugging Face weights with `convert_hf_to_gguf.py --mtp --outtype q8_0`. Then re-quantize it to Q4_K_M with a Q4_K output tensor (for example `llama-quantize --allow-requantize --output-tensor-type q4_k HEAD-q8_0.gguf HEAD.gguf Q4_K_M`). The head's output projection is most of its cost, so the smaller head is faster overall.

### Plugin options (`--spec-plugin-args`)

| Key | Meaning |
|---|---|
| `table=FILE` | The lookup table (required) |
| `reranker=FILE` | Pick among the proposals with this reranker |
| `pad=0\|1` | Always draft `n_max` tokens, so the batch shape never changes (default 0; use 1) |
| `tau=P` | Without `pad`: keep drafting while the chain's chance of full acceptance is at least P (default 0.3) |
| `n_max=N` | Longest draft (default: whatever the host allows) |
| `source=both\|table\|context` | Which lookups to use (default both) |
| `vocab_hash=HEX` | Refuse to load unless the model's vocabulary matches. `python plugin/vocab_hash.py MODEL.gguf` prints the hash. |

## Building a table

A table is counted from text the target model wrote itself. Its own output is worth far more per token than anyone else's. For code, the steps are:

1. **Generate.** `tools/gen_code.py` has a llama-server answer OSS-Instruct coding prompts (`ise-uiuc/Magicoder-OSS-Instruct-75K`) and stores the answers as token IDs. The shipped table used about 74,000 answers (39.5M tokens): 5,000 answered on a CPU and 69,000 on one rented A40 in 14 hours.
2. **Count and calibrate.** `tools/code_study.py --export` counts the table, calibrates its confidences on held-out answers and writes a `.mdt` file. `tools/build_table.py` does the same for general text. `--check` on either replays the C++ drafter against the Python one.
3. **Reranker (optional, about +5% tokens per step).** The reranker is trained with `research/reranker/rerank7.py`. `tools/export_reranker.py` then writes the final pair of files: the table in its version 2 format and the reranker. It also checks that the C++ reranker picks the same token as the Python one (it did at 99.975% of positions). Training is still research code; see [research/](research/).

The shipped files also contain 9B chat and math text from another project, which is not published. Without it, `code_study.py --skip-shards` builds a code-only table.

## Benchmarking

`bench/run_plugin_live.sh` produced the results table. It runs plain decoding and each drafter setup on the same prompts, in turns of at most 9 minutes, only while no other process is on the GPU. `tools/show_wide.py` then summarizes the turns with confidence intervals.

The script wraps each turn in `arbiter run …`, a local scheduler for shared GPUs; remove that prefix on another machine. The prompt sets are in [`bench/prompts/`](bench/prompts/).

## Repository layout

| Path | Contents |
|---|---|
| `src/memdraft.h` | The drafter: table, request index, reranker, chain rule. Header-only C++17, with no llama.cpp types. |
| `plugin/` | The llama.cpp plugin, its build script, `vocab_hash.py`, and example plugins (`ngram-mod`, a draft model, an MTP head) with equivalence tests |
| `patches/llama-plugin-hook.patch` | Adds `--spec-type plugin` to llama.cpp `edec462` |
| `tools/` | Generating training text, building and checking tables, the benchmark client |
| `bench/` | The benchmark script, prompt sets and results |
| `tests/` | `python -m pytest tests` |
| `docs/history.md` | How the project got here, including the failed first attempt |
| `research/` | Experiments that did not ship, or that led to the shipped setup, with their raw results |

Large files live outside git. `data/` holds tables and generated rows. `$MEMDRAFT_CACHE` (default `~/.cache/model-experiments/memdraft`) holds llama.cpp builds, GGUFs, and the tokenizer and prompt dataset. Python dependencies are listed in `requirements.txt`.

## License

MIT; see [LICENSE](LICENSE).
