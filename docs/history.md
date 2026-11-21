# How MemDraft got here

MemDraft started as one idea and became another. Speeds are given as generated tokens per second, with the speedup over plain decoding on the same prompts in parentheses. This page tells the story in order: what each step tried, what it found, and why the next step followed. Raw results are in [`research/results/`](../research/results/) and [`bench/results/`](../bench/results/). Every timing below is Qwen3.5-9B Q4_K_M on an AMD Radeon 8060S (Vulkan), or on its CPU where that is stated, with greedy decoding.

| When (2026) | Step | Outcome |
|---|---|---|
| Oct 25 – Nov 7 | First attempt: a cache of past continuations, in front of DFlash2 | Failed; deleted |
| Nov 8 – 10 | Restart: a lookup table built from the model's own text | 40.2 tok/s (1.24×) on mixed prompts (GPU pilot) |
| Nov 10 – 12 | Tables built from the model's own code, plus a reranker | 65.0 tok/s (1.93×) on coding prompts |
| Nov 15 – 17 | Wider GPU runs: HumanEval, relaxed acceptance, llama.cpp's drafters, the MTP head | `ngram-simple` in front: 124.6 tok/s (3.27×) on HumanEval |
| Nov 18 | The MTP head drafts the first token, MemDraft the rest | 82.4 tok/s (2.16×) coding, 137.4 tok/s (3.60×) HumanEval |
| Nov 18 – 19 | MemDraft becomes a llama.cpp plugin | Same speed as the patched build |
| Nov 15 | Side experiment: Qwen3.8 n-gram embeddings as reranker features | Built, never measured |

## 1. The first attempt: a continuation cache (failed)

**Idea.** Keep a table of token continuations seen earlier in the request stream, filled at run time. Put it in front of DFlash2, a strong learned drafter, so that repeated text could be drafted cheaply. Keys were exact 4- or 8-token sequences, drafts up to 16 tokens, and a small MLP decided when to use the cache.

**What happened.** On new requests the cache barely helped. With Qwen3.8-27B on CPU, tokens accepted per proposal went from 3.37 with DFlash2 alone to 3.39 to 3.53 with the cache. Acceptance fell, and the one timed comparison was slower: 127.8 s of server time against 122.8 s (0.96×). That run recorded server time, not tokens per second. It only paid off on exact repeats. On Qwen3.6-35B-A3B, every n-gram source placed in front of DFlash2 lowered tokens per step, MemDraft included. No clean timing was ever obtained, because every timing run got stuck in a queue on a shared machine.

**Why it failed.** Long drafts from a cache are usually wrong on new text, and each rejected token still costs verification time. DFlash2 was already good, and similar published work (Cacheback, SuffixDecoding, ReSpec) already covers the idea. The code was deleted on 2026-11-08.

**What carried over:**

- Speculative decoding in llama.cpp is not bit-exact. Greedy output depends on batch size, so speculative setups must be compared on task quality, not on identical text.
- Acceptance rate is not speed.
- A speedup on a tiny model means nothing.
- Measure on a quiet machine, as one job, planned from the start.

## 2. The restart: a table built ahead of time (Nov 8–10)

**New question.** Can a drafter with no draft model at all beat plain decoding in tokens per second, standing alone?

**Why it might.** A decode step with `n` draft tokens costs about `1 + β·n` plain steps, where β is the cost of one more token in the batch. On the 9B through Vulkan, β is about 0.065. When drafting is free, a guess is worth sending whenever its chance of being right is above about β. A guess right only one time in five still pays.

**The drafter.** For each of the last 1 to 6 tokens of context, a table counted from text the model wrote stores its most likely next token and a calibrated confidence. A second lookup does the same over the current request. A draft step takes about 2 µs.

**Offline study** (`tools/ngram_study.py`; results in `research/results/study-9b-*.json`). This used 35M tokens the 9B wrote for math and chat prompts, with a held-out test set:

- Using both lookups beat either one alone, because the table covers chat and the request covers math.
- The model's plain output is enough to build the table; no extra model pass is needed.
- Contexts seen fewer than three times can be dropped: 64 MiB loses almost nothing against 512 MiB.
- A table carries poorly outside the kind of text it was counted from.

**Two hidden costs in llama.cpp,** both found here and both hitting short drafts hardest:

1. **Rollback.** Qwen3.5 has recurrent layers. With only n-gram drafters enabled, llama.cpp saved a checkpoint before each drafted step and, after any rejection, restored it and decoded again. The patch asks for per-token snapshots instead, as model drafters get.
2. **Graph rebuilds.** llama.cpp rebuilds its compute graph whenever the batch size changes, which costs 5.7 ms on the 9B, as much as three draft tokens. Variable-length drafts change it at most steps. The fix is `pad`: always draft the same number of tokens. This one change turned a slowdown on chat into a gain.

**First live numbers** (pilot, busy machine):

- On the GPU with mixed prompts, against 32.4 tok/s plain, two-token drafts gave 40.2 tok/s (1.24×). By kind of prompt: 35.3 tok/s (1.09×) on chat, 38.1 tok/s (1.17×) on code, 43.3 tok/s (1.34×) on editing and 46.0 tok/s (1.42×) on math.
- On CPU, against 5.88 tok/s plain, MemDraft gave 6.95 tok/s (1.18×) and llama.cpp's `ngram-cache` 5.85 tok/s (0.99×).
- The β cost model predicted these within 4%.

## 3. Code, and the model's own code (Nov 10–12)

The first table had no code in it. Coding prompts turned out to suit lookup drafting well anyway, because code repeats its own names: the request lookup alone got 37% of first guesses right on code, against 13% on chat.

**The model's own code is worth far more than anyone else's.** The 9B answered 800 OSS-Instruct prompts. Offline:

- 0.22M tokens of its own code did better than the 19.5M-token general table.
- gpt-3.5's solutions to twice as many prompts did worse than the 9B's own code.

On CPU, against 8.50 tok/s plain, a table with code in it gave 11.62 tok/s (1.37×), and the general table 10.32 tok/s (1.21×).

**Scaling up.** The 9B then answered every remaining OSS-Instruct prompt: 4,200 on the local CPU and 69,000 on one rented A40 (37M tokens in 14 hours, $8.71). Offline, each doubling of own code in the table was still helping at 39M tokens; the end of that curve has not been reached.

**A reranker.** Instead of taking the most confident proposal, a small MLP scores each proposed token using the lookups' statistics and the target's token embeddings. It added about 6% tokens per step offline. It was trained in [`research/reranker/`](../research/reranker/), and its C++ version matches the Python one on 99.975% of positions.

**First GPU result on code** (12 prompts, one round, against 33.7 tok/s plain): 60.9 tok/s (1.81×) with the bigger table, and **65.0 tok/s (1.93×)** with the reranker drafting three tokens. On the GPU a drafted step costs about what a plain step costs, so speed tracks tokens per step almost one for one.

## 4. Wider GPU runs (Nov 15–17)

With GPU turns now taken only when the GPU was idle and the machine quiet, the runs became cleaner and larger.

**HumanEval.** On all 164 problems the table and reranker gave **87.4 tok/s (2.29×)** against 38.1 tok/s plain. On the coding prompts they gave 68.6 tok/s (1.80×).

**Relaxed acceptance was not worth it.** A patch option kept a drafted token whenever the model gave it at least R times the probability of its own top choice. Lowering R from 1.0 to 0.1:

- did not speed up HumanEval: 87.4 tok/s (2.29×) became 86.7 tok/s (2.27×);
- lowered pass@1 from 83.5% to 80.5% (plain decoding: 84.8%).

MemDraft stays lossless.

**Against llama.cpp's own lookup drafters**, run as shipped (36 HumanEval problems, 12 coding prompts):

| Drafter | Coding prompts | HumanEval | Whole-file edits |
|---|---:|---:|---:|
| Plain decoding | 38.0 tok/s (1.00×) | 38.1 tok/s (1.00×) | 36.1 tok/s (1.00×) |
| `ngram-cache` | 37.3 tok/s (0.98×) | 50.1 tok/s (1.31×) | 64.7 tok/s (1.79×) |
| `ngram-mod` | 40.5 tok/s (1.07×) | 65.7 tok/s (1.72×) | 86.5 tok/s (2.39×) |
| `ngram-simple` | 43.1 tok/s (1.14×) | 74.3 tok/s (1.95×) | 101.7 tok/s (2.82×) |
| MemDraft, 3 tokens | 68.7 tok/s (1.81×) | 91.1 tok/s (2.39×) | 89.7 tok/s (2.48×) |
| **`ngram-simple`, then MemDraft** | **69.8 tok/s (1.83×)** | **124.6 tok/s (3.27×)** | **139.1 tok/s (3.85×)** |

The two kinds of drafter complement each other. `ngram-simple` drafts long runs when the answer copies from the prompt, and MemDraft covers everything else. Chaining them became the default. Longer MemDraft drafts did not help on the coding prompts. 5 tokens gave 69.1 tok/s (1.82×), about even with 3, and 8 tokens gave 37.5 tok/s (0.99×), slower than plain decoding.

**The model's own MTP head.** Qwen3.5 ships a multi-token-prediction head. Run alone through llama.cpp's `draft-mtp`, it gave 63.0 tok/s (1.66×) on code and 73.3 tok/s (1.93×) on HumanEval, both below MemDraft. It accepts more tokens per step, but each step costs 1.9 plain steps because the head is a real model.

## 5. The head drafts first (Nov 18)

The head is good at the next token, and MemDraft is cheap for the tokens after it. So the head now drafts one token and MemDraft continues from there. The head's output projection (248k × 4096) is most of its cost, so smaller quantizations were tried:

| Head | Coding prompts | HumanEval |
|---|---:|---:|
| Plain decoding | 38.1 tok/s (1.00×) | 38.2 tok/s (1.00×) |
| none (`ngram-simple`, then MemDraft) | 70.1 tok/s (1.84×) | 120.3 tok/s (3.15×) |
| Q8_0 head, then MemDraft | 77.1 tok/s (2.02×) | 129.0 tok/s (3.38×) |
| Q4_K_M head with Q4_K output (572 MB), then MemDraft | **82.4 tok/s (2.16×)** | **137.4 tok/s (3.60×)** |

The cheapest head did best. This is the setup in the README.

## 6. A plugin instead of a patch (Nov 18–19)

Until now MemDraft lived inside a patched llama.cpp ([`research/patched-build/`](../research/patched-build/)). The patch was tied to one commit and touched llama.cpp's drafter code. The replacement is a small hook ([`patches/llama-plugin-hook.patch`](../patches/llama-plugin-hook.patch)): `--spec-type plugin` loads any drafter from a shared library, and `--spec-plugin-prefix` lets a draft model such as the MTP head go first.

**Equivalence.** MemDraft as a plugin ran at the same speed as the patched build on coding prompts. Behind `ngram-simple` it gave 70.5 tok/s (1.85×) against 70.1 tok/s (1.84×), and with the head 81.6 tok/s (2.14×) against 81.8 tok/s (2.15×).

**Example plugins.** To check that the hook is general, three of llama.cpp's built-in drafters were rewritten as plugins: `ngram-mod`, a separate draft model and the MTP head.

- The `ngram-mod` plugin drafts the same tokens as the built-in one and ran at the same speed: 41.1 tok/s (1.08×) against 41.0 tok/s (1.08×).
- The draft-model plugin was slightly faster than the built-in: 47.7 tok/s (1.26×) against 44.6 tok/s (1.17×).
- The MTP plugin was slower at first: 64.2 tok/s (1.69×) against the built-in's 73.9 tok/s (1.94×). The cause was taking the argmax over the full vocabulary, at 0.68 ms per call. A top-10 selection, at 0.04 ms, brought it to 71.5 tok/s (1.88×) against 73.9 tok/s (1.95×).

## Side experiment: Qwen3.8 n-gram embeddings (Nov 15, unfinished)

Qwen3.8-Flash-Next keeps learned embeddings for hashed n-grams. The idea was to feed those rows to the reranker as extra features. The feature extraction, the training code, a native scorer and head-selection tools are all built and tested ([`research/ple/`](../research/ple/), [`research/qwen38-head-selection/`](../research/qwen38-head-selection/)). The experiment was never run, because the machine did not have the model's weights or enough free memory.

## Still open

- A clean run on a quiet machine, with more prompts.
- Other models. The table should work for any model with the Qwen3.5 tokenizer (the 27B, 35B-A3B and 2B), but this has not been measured.
- Chat. The table in the results is built for code, and chat was the weakest domain in the general study.
- Where the own-code curve flattens, using prompts beyond OSS-Instruct.
