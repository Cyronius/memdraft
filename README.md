# MemDraft

MemDraft is a speculative-decoding method for llama.cpp that doesn't need a separate draft model. It proposes the next tokens from a lookup table, and the target model checks the guesses as in any speculative decoding. A draft step costs microseconds, so even guesses that are right only some of the time make decoding faster.

The code is on GitHub at **[Cyronius/memdraft](https://github.com/Cyronius/memdraft)**. It builds as a plugin, `libmemdraft.so`, loaded by llama.cpp's `--spec-type plugin`. That option comes from a small patch to llama.cpp in the GitHub repo.

MemDraft needs a drafter made for your model: a lookup table (`.mdt`) and, optionally, a reranker (`.mdr`). MemDraft does not build them while serving. They are built once, offline: the target model first answers many prompts, then scripts count those answers into a table and train the reranker. The drafter behind the results below took about 40M generated tokens, most of them from 14 hours on one rented A40.

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

- **Compatibility.** The table works with any model that uses the same tokenizer: the Qwen3.5, 3.6 and 3.8 family, 248,070 tokens. `vocab_hash` makes the plugin refuse any other tokenizer. The table has only been measured with the 9B. The reranker was trained against the 9B; it loads with other models in the family, but it has not been measured there.
- **Offline, on held-out answers.** Drafting 3 tokens: 2.01 tokens per step with the reranker and 1.86 without it. At plain decoding's 38.0 tok/s and the 9B's measured GPU step costs, that predicts 62.7 tok/s (1.65×) and 58.2 tok/s (1.53×).

## How it works

At each step MemDraft proposes tokens from two lookups:

1. **The table.** For each of the last 1 to 6 tokens of context, it stores the token the model most often wrote next and how often that guess was right on held-out text. MemDraft loads it at startup and only reads it.
2. **The current request.** The same kind of lookup over the prompt and the tokens generated so far, built up as the request runs.

An optional reranker picks among the proposals. It is a small MLP over the proposals and the target's token embeddings, trained offline alongside the table and stored in a `.mdr` file. MemDraft always drafts the same number of tokens (`pad=1`), so llama.cpp's compute graph keeps its shape and is never rebuilt. A draft call takes about 60 µs with the reranker and 3 µs without it.

It works best behind llama.cpp's `ngram-simple` drafter. `ngram-simple` also needs no draft model, and no table either: when the last 12 tokens appeared earlier in the same request, it drafts up to 48 tokens that followed that earlier match. It catches long copies, such as an answer that repeats code from the prompt. When it finds no match, MemDraft drafts. Optionally, the model's own multi-token prediction (MTP) head drafts the first token and MemDraft continues from it. The MTP head is a small extra layer that some models ship with (Qwen3.5 does) to predict tokens past the next one.

## Results

These numbers are for Qwen3.5-9B Q4_K_M on an AMD Radeon 8060S through Vulkan, with greedy decoding and no other process on the GPU, using the [Qwen3.5-9B drafter](#the-qwen35-9b-drafter). The coding prompts below are held-out OSS-Instruct prompts: no table holds answers to them, and the reranker never saw them.

- **Coding prompts:** 12 OSS-Instruct prompts, 2 rounds.
- **HumanEval:** problems 0–11.
- Each figure is generated tokens per second, with the speedup over plain decoding on the same prompts in parentheses.

| Setup | Coding prompts | HumanEval |
|---|---:|---:|
| Plain decoding | 38.0 tok/s (1.00×) | 38.2 tok/s (1.00×) |
| **MemDraft alone**¹ | **68.7 tok/s (1.81×)** | **91.1 tok/s (2.39×)** |
| `ngram-simple` alone¹ | 43.1 tok/s (1.14×) | 74.3 tok/s (1.95×) |
| **`ngram-simple`, then MemDraft** | **70.5 tok/s (1.85×)** | **125.5 tok/s (3.29×)** |
| The MTP head alone (`draft-mtp`, 3 tokens) | 73.9 tok/s (1.94×) | 88.0 tok/s (2.30×) |
| **`ngram-simple`, then the MTP head, then MemDraft** | **81.6 tok/s (2.14×)** | **136.3 tok/s (3.57×)** |

Rows without a mark are from 2026-11-18. ¹ From 2026-11-17, in the earlier patched build of llama.cpp, with one round of the coding prompts and HumanEval problems 0–35. The patched build and the plugin ran at the same speed for `ngram-simple`, then MemDraft: 70.1 tok/s (1.84×) and 70.5 tok/s (1.85×).

Two other drafters were also tested and did worse than every setup above:

- **`ngram-mod`**, the lookup drafter that llama.cpp's `--spec-default` turns on: 41.0 tok/s (1.08×) on coding prompts, and 67.8 tok/s (1.78×) on HumanEval in a run on 2026-11-19.
- **Qwen3.5-2B as a separate draft model**, 3 tokens: 44.6 tok/s (1.17×) on coding prompts. On HumanEval only 2 problems completed, at 51.7 tok/s (1.36×).

What the results show:

- **On the coding prompts, nearly all the gain is MemDraft's**: 68.7 tok/s (1.81×) alone, 70.5 tok/s (1.85×) behind `ngram-simple`. On HumanEval, `ngram-simple` adds the long copies, taking MemDraft from 91.1 tok/s (2.39×) to 125.5 tok/s (3.29×). HumanEval answers often restate code from the prompt, so the coding prompts are the more typical case.
- **The MTP head alone beats MemDraft alone on the coding prompts** (73.9 tok/s (1.94×) against 68.7 tok/s (1.81×)). It accepts more tokens per step, but each step costs more because the head is a real model. Putting the head first and MemDraft after it is fastest.
- With both chained MemDraft setups, every prompt in every round ran faster than plain decoding.
- **Verification is lossless, but llama.cpp's batched decoding is not bit-exact.** The output text matched plain decoding on 11 of 12 HumanEval problems and on 8 of 24 coding-prompt runs. Where it differs, it splits at a near-tie between two tokens; plain decoding also changes this way when only the batch size changes.
- **Answer quality held up on HumanEval.** Over all 164 problems, pass@1 was 83.5% against 84.8% for plain decoding: two problems flipped. That run was MemDraft alone (87.4 tok/s (2.29×) over the 164 problems, against 38.1 tok/s plain), with the same table and reranker, in the earlier patched build of llama.cpp.
- **The setup is narrow so far.** It has been measured on one model, on code, on one GPU. On a mix of chat, math and editing prompts, an earlier table built from the 9B's chat and math text gave 40.2 tok/s (1.24×) overall against 32.4 tok/s plain, but only 35.3 tok/s (1.09×) on chat.
- Raw results are in [`bench/results/`](https://github.com/Cyronius/memdraft/tree/main/bench/results); the rows marked ¹ are in [`research/results/prior-9b-vk-20261117-1614/`](https://github.com/Cyronius/memdraft/tree/main/research/results/prior-9b-vk-20261117-1614). How MemDraft got here, including what failed, is in [`docs/history.md`](https://github.com/Cyronius/memdraft/blob/main/docs/history.md).

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

- **Code:** MIT; see [LICENSE](https://github.com/Cyronius/memdraft/blob/main/LICENSE).
- **Drafter files on Hugging Face:** Apache 2.0, the license of the Qwen3.5-9B weights they derive from.
