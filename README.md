# MemDraft

MemDraft is a speculative-decoding method for llama.cpp that doesn't need a separate draft model. It proposes the next tokens from a lookup table, and the target model checks the guesses as in any speculative decoding. A draft step costs microseconds, so even guesses that are right only some of the time make decoding faster.

The code is on GitHub at **[Cyronius/memdraft](https://github.com/Cyronius/memdraft)**. It builds as a plugin, `libmemdraft.so`, loaded by llama.cpp's `--spec-type plugin`. That option comes from a small patch to llama.cpp in the GitHub repo.

MemDraft needs a drafter made for your model: a lookup table (`.mdt`) and, optionally, a reranker (`.mdr`). MemDraft does not build them while serving. They are built once, offline: the target model first answers many prompts, then scripts count those answers into a table and train the reranker. The drafter behind the results below took about 40M generated tokens, most of them from 14 hours on one rented A40.

## Results

One drafter, built from Qwen3.5-9B's own text, used unchanged on four Qwen3.5 sizes:

| Model | Plain decoding | Coding prompts | HumanEval |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 244 tok/s | 451 tok/s (1.85×) | 626 tok/s (2.54×) |
| Qwen3.5-2B | 136 tok/s | 248 tok/s (1.83×) | 352 tok/s (2.58×) |
| Qwen3.5-4B | 65 tok/s | 110 tok/s (1.69×) | 199 tok/s (3.04×) |
| Qwen3.5-9B | 38 tok/s | 70 tok/s (1.84×) | 113 tok/s (2.96×) |

- **Setup:** `ngram-simple`, then MemDraft with the reranker, drafting 3 tokens. Q4_K_M models on an AMD Radeon 8060S through Vulkan, greedy decoding, 2026-11-21.
- **Prompts:** 12 held-out OSS-Instruct coding prompts (2 rounds) and HumanEval problems 0–11. No table holds answers to them. The speedup is over plain decoding on the same prompts; plain decoding runs within about 1% of the same speed on both sets.
- **Output:** the target model checks every drafted token, so the text is the target's own. See [Quality](#quality).
- **Limits:** Qwen3.5 only, code only, one GPU. The 27B and 35B-A3B are not measured yet. On chat, an earlier table gave only 1.09×.

## Drafters

Ready-made drafters are on Hugging Face at **[Cyronius/memdraft](https://huggingface.co/Cyronius/memdraft)**, one folder per target model. For any other model, [build one](#building-a-table).

| Target model | Domain | Folder | Table | Reranker | Vocabulary hash |
|---|---|---|---:|---:|---|
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Code | [`qwen3.5-9b/`](https://huggingface.co/Cyronius/memdraft/tree/main/qwen3.5-9b) | 256 MiB | 124 MiB | `c52114e74bc3a20f` |

| File | SHA-256 |
|---|---|
| `qwen3.5-9b/code.mdt` | `5d835a951a03e86f3cda826738e14f3ea1d38b06487f567a5afe80af7eebf3db` |
| `qwen3.5-9b/code.mdr` | `6458dcf9b905fb0b28338b19abb99a6ce42b73681063f83858860fd234c84fd7` |

### The Qwen3.5-9B drafter

The table counts 77.5M tokens of text, keeping contexts seen at least 3 times (14.8M entries):

- **Code the 9B wrote:** answers to about 74,000 prompts from [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K), at temperature 0 with thinking off.
- **That dataset's own gpt-3.5 solutions**, except for the held-out prompts.
- **Chat and math text the 9B wrote:** answers to prompts from GSM8K, MATH, OpenThoughts and UltraChat.

The reranker was fine-tuned on positions from the 9B's code. It includes 64-wide projections of the 9B's token embeddings.

- **Compatibility.** The table works with any model that uses the same tokenizer: the Qwen3.5, 3.6 and 3.8 family, 248,070 tokens. `vocab_hash` makes the plugin refuse any other tokenizer. It has been measured with Qwen3.5 0.8B, 2B, 4B and 9B (see [Results](#results)), and the reranker helps on all four, although it was trained against the 9B. The 27B and 35B-A3B have not been measured.

## How it works

At each step MemDraft proposes tokens from two lookups:

1. **The table.** For each of the last 1 to 6 tokens of context, it stores the token the model most often wrote next and how often that guess was right on held-out text. MemDraft loads it at startup and only reads it.
2. **The current request.** The same kind of lookup over the prompt and the tokens generated so far, built up as the request runs.

An optional reranker picks among the proposals. It is a small MLP over the proposals and the target's token embeddings, trained offline alongside the table and stored in a `.mdr` file. MemDraft always drafts the same number of tokens (`pad=1`), so llama.cpp's compute graph keeps its shape and is never rebuilt. A draft call takes about 60 µs with the reranker and 3 µs without it.

It can run behind llama.cpp's `ngram-simple` drafter. `ngram-simple` also needs no draft model, and no table either: when the last 12 tokens appeared earlier in the same request, it drafts up to 48 tokens that followed that earlier match. It catches long copies, such as an answer that repeats code from the prompt. When it finds no match, MemDraft drafts. This pays when answers copy from the prompt (HumanEval); on ordinary code, MemDraft alone is as fast. Optionally, the model's own multi-token prediction (MTP) head drafts the first token and MemDraft continues from it. The MTP head is a small extra layer that some models ship with (Qwen3.5 does) to predict tokens past the next one.

## Why a lookup drafter

A drafter helps only when it is cheap next to the target. The speedup is about the tokens accepted per step divided by the cost of a step, so a drafter that guesses well but runs slowly gives back its gain. On the coding prompts with Qwen3.5-9B, with step cost relative to one plain decoding step:

| Drafter (3 tokens) | Tokens per step | Step cost | Speed |
|---|---:|---:|---:|
| Qwen3.5-2B as a separate draft model | 2.77 | 2.36× | 44.6 tok/s (1.17×) |
| The 9B's MTP head (`draft-mtp`) | 3.08 | 1.59× | 73.9 tok/s (1.94×) |
| `ngram-simple`, then MemDraft | 2.19 | 1.18× | 70.5 tok/s (1.85×) |

The 2B draft model gets more tokens per step than MemDraft, but it is the slowest setup. A MemDraft draft call takes 3 to 60 µs, against about 26 ms for one plain step of the 9B. Most of MemDraft's 1.18× step cost is the target checking 4 tokens instead of 1.

- **Smaller targets.** A smaller target takes less time per step, so a drafter that runs a model takes a larger share of each step. MemDraft's cost stays in microseconds at any target size. On Qwen3.5-0.8B, at 244 tok/s plain, MemDraft still gives 1.85× on code. Its step cost rises from 1.19× on the 9B to 1.38×; a draft model's step cost (2.36× on the 9B) would leave nothing.
- **No VRAM.** The table and reranker load into system RAM (380 MiB for the Qwen3.5-9B drafter) and run on the CPU. A draft model needs VRAM for its weights and its own KV cache, and in the results the MTP head also ran on the GPU. The target still checks MemDraft's guesses on the GPU.

## Detailed results

Settings and prompts are as in [Results](#results). Raw results are in [`bench/results/`](https://github.com/Cyronius/memdraft/tree/main/bench/results); `bench/run_qwen_family.sh` reruns the size comparison.

### Which setup to use

Speedup over plain decoding, all sizes, 2026-11-21:

| Model | MemDraft alone | `ngram-simple`, then MemDraft | The same, no reranker |
|---|---:|---:|---:|
| **Coding prompts** | | | |
| 0.8B | 1.78× | 1.85× | 1.69× |
| 2B | 1.82× | 1.83× | 1.67× |
| 4B | 1.70× | 1.69× | 1.61× |
| 9B | 1.82× | 1.84× | 1.74× |
| **HumanEval** | | | |
| 0.8B | 2.05× | 2.54× | 2.59× |
| 2B | 2.22× | 2.58× | 2.46× |
| 4B | 2.24× | 3.04× | 2.89× |
| 9B | 2.31× | 2.96× | 3.09× |

- **`ngram-simple` pays only when answers copy from the prompt.** On the coding prompts, MemDraft alone ties the chain on the 2B, 4B and 9B. On HumanEval, whose answers often restate code from the prompt, the chain is well ahead at every size.
- **The reranker pays on code at every size**, by 5% to 10%. On HumanEval it helps the 2B and 4B and loses a little on the 0.8B and 9B, all within the 95% intervals.
- **Smaller models accept more but pay more per step.** On the coding prompts, the chain's acceptance (draft tokens accepted over draft tokens proposed) is 41.7%, 37.2%, 33.4% and 35.2% from 0.8B to 9B. Its step cost, in plain decoding steps, is 1.38×, 1.27×, 1.27× and 1.19×: the target's check of 4 tokens weighs more on a fast model, as [the cost model](#why-a-lookup-drafter) predicts.
- **CPU load matters.** MemDraft runs on the CPU and plain decoding on the GPU, so a busy CPU slows only MemDraft. A first 4B run at a load average of 8 gave 1.56× on code; the rerun on a quiet machine, with the same accepted tokens, gave 1.69×. Turns started at a load average of 0.1 to 3.7, except the 9B coding turns (5.3 to 7.0, yet plain decoding and the chain both matched 2026-11-18), the 9B HumanEval turn (4.4) and the 4B HumanEval turn (5.3).

### Against other drafters (Qwen3.5-9B)

| Setup | Coding prompts | HumanEval |
|---|---:|---:|
| Plain decoding | 38.0 tok/s (1.00×) | 38.2 tok/s (1.00×) |
| MemDraft alone¹ | 69.4 tok/s (1.82×) | 88.2 tok/s (2.31×) |
| `ngram-simple` alone¹ | 43.3 tok/s (1.13×) | 73.7 tok/s (1.93×) |
| `ngram-simple`, then MemDraft | 70.5 tok/s (1.85×) | 125.5 tok/s (3.29×) |
| The MTP head alone (`draft-mtp`, 3 tokens) | 73.9 tok/s (1.94×) | 88.0 tok/s (2.30×) |
| **`ngram-simple`, then the MTP head, then MemDraft** | **81.6 tok/s (2.14×)** | **136.3 tok/s (3.57×)** |
| `ngram-mod`² | 41.0 tok/s (1.08×) | 67.8 tok/s (1.78×) |
| Qwen3.5-2B as a separate draft model, 3 tokens³ | 44.6 tok/s (1.17×) | 51.7 tok/s (1.36×) |

Rows without a mark are from 2026-11-18. ¹ From 2026-11-21, the run in [Results](#results), where `ngram-simple`, then MemDraft gave 70.3 tok/s (1.84×) on coding prompts and 113 tok/s (2.96×) on HumanEval. ² The lookup drafter that llama.cpp's `--spec-default` turns on; HumanEval from 2026-11-19. ³ On HumanEval only 2 problems completed.

- **The MTP head alone beats MemDraft alone on the coding prompts** (1.94× against 1.82×). It accepts more tokens per step, but each step costs more because the head is a real model. Putting the head first and MemDraft after it is fastest.
- With both chained MemDraft setups, every prompt in every round ran faster than plain decoding.

### Quality

- **Verification is lossless, but llama.cpp's batched decoding is not bit-exact.** On the 9B, the output text matched plain decoding on 11 of 12 HumanEval problems and on 8 of 24 coding-prompt runs. Where it differs, it splits at a near-tie between two tokens; plain decoding also changes this way when only the batch size changes.
- **Answer quality held up on HumanEval.** Over all 164 problems with the 9B, pass@1 was 83.5% against 84.8% for plain decoding: two problems flipped. That run was MemDraft alone (87.4 tok/s (2.29×) over the 164 problems, against 38.1 tok/s plain), with the same table and reranker, in an earlier patched build of llama.cpp.
- **Chat is weak.** On a mix of chat, math and editing prompts, an earlier table built from the 9B's chat and math text gave 40.2 tok/s (1.24×) overall against 32.4 tok/s plain, but only 35.3 tok/s (1.09×) on chat.

How MemDraft got here, including what failed, is in [`docs/history.md`](https://github.com/Cyronius/memdraft/blob/main/docs/history.md).

## Quick start

You need:

- llama.cpp at commit `edec462`
- the [MemDraft code](https://github.com/Cyronius/memdraft)
- a drafter for your model: a table (`.mdt`) and, optionally, a reranker (`.mdr`). See [Drafters](#drafters).

The commands below use the Qwen3.5-9B drafter from the results.

```bash
# 0. Get MemDraft and download the drafter
git clone https://github.com/Cyronius/memdraft
hf download Cyronius/memdraft --include 'qwen3.5-9b/*' --local-dir drafters

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
  --spec-plugin-args table=drafters/qwen3.5-9b/code.mdt,reranker=drafters/qwen3.5-9b/code.mdr,pad=1,vocab_hash=c52114e74bc3a20f
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

1. **Generate.** `tools/gen_code.py` has a llama-server answer OSS-Instruct coding prompts (`ise-uiuc/Magicoder-OSS-Instruct-75K`) and stores the answers as token IDs. The table in the results used about 74,000 answers (39.5M tokens): 5,000 answered on a CPU and 69,000 on one rented A40 in 14 hours.
2. **Count and calibrate.** `tools/code_study.py --export` counts the table, calibrates its confidences on held-out answers and writes a `.mdt` file. `tools/build_table.py` does the same for general text. `--check` on either replays the C++ drafter against the Python one.
3. **Reranker (optional, about +5% tokens per step).** The reranker is trained with `research/reranker/rerank7.py`. `tools/export_reranker.py` then writes the final pair of files: the table in its version 2 format and the reranker. It also checks that the C++ reranker picks the same token as the Python one (it did at 99.975% of positions). Training is still research code; see [`research/`](https://github.com/Cyronius/memdraft/tree/main/research).

The published Qwen3.5-9B table also contains 9B chat and math text from another project; that text itself is not published. Without it, `code_study.py --skip-shards` builds a code-only table.

This README is also the Hugging Face model card. To add a drafter for another model:

1. Upload its `.mdt` and `.mdr` to a new folder in the Hugging Face repo.
2. Add a row to [Drafters](#drafters), with the SHA-256 of both files.
3. Add the model under `base_model` in `hf/card-metadata.yml`.
4. Publish the card: `bash hf/publish-card.sh`.

## Benchmarking

`bench/run_plugin_live.sh` produced the results table. It runs plain decoding and each drafter setup on the same prompts, in turns of at most 9 minutes, only while no other process is on the GPU. `tools/show_wide.py` then summarizes the turns with confidence intervals.

The script wraps each turn in `arbiter run …`, a local scheduler for shared GPUs; remove that prefix on another machine. The prompt sets are in [`bench/prompts/`](https://github.com/Cyronius/memdraft/tree/main/bench/prompts).

## Repository layout

| Path | Contents |
|---|---|
| `src/memdraft.h` | The drafter: table, request index, reranker, chain rule. Header-only C++17, with no llama.cpp types. |
| `plugin/` | The llama.cpp plugin, its build script, `vocab_hash.py`, and example plugins (`ngram-mod`, a draft model, an MTP head) with equivalence tests |
| `patches/llama-plugin-hook.patch` | Adds `--spec-type plugin` to llama.cpp `edec462` |
| `tools/` | Generating training text, building and checking tables, the benchmark client |
| `bench/` | The benchmark script, prompt sets and results |
| `hf/` | The Hugging Face card metadata and the script that publishes this README as the card |
| `tests/` | `python -m pytest tests` |
| `docs/history.md` | How the project got here, including the failed first attempt |
| `research/` | Experiments that did not ship, or that led to the shipped setup, with their raw results |

Large files live outside git. `data/` holds tables and generated rows. `$MEMDRAFT_CACHE` (default `~/.cache/model-experiments/memdraft`) holds llama.cpp builds, GGUFs, and the tokenizer and prompt dataset. Python dependencies are listed in `requirements.txt`.

## License

- **Code:** MIT or Apache 2.0, at your option; see [LICENSE-MIT](https://github.com/Cyronius/memdraft/blob/main/LICENSE-MIT) and [LICENSE-APACHE](https://github.com/Cyronius/memdraft/blob/main/LICENSE-APACHE).
- **Drafter files on Hugging Face:** Apache 2.0, the license of the Qwen3.5-9B weights they derive from.
