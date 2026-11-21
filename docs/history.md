# How MemDraft got here

MemDraft started as one idea and became another. This page tells the story in order: what each step tried, what it found, and why the next step followed. Raw results are in [`research/results/`](../research/results/) and [`bench/results/`](../bench/results/). Every timing below is Qwen3.5-9B Q4_K_M on an AMD Radeon 8060S (Vulkan), or on its CPU where that is stated, with greedy decoding.

| When (2026) | Step | Outcome |
|---|---|---|
| Oct 25 – Nov 7 | First attempt: a cache of past continuations, in front of DFlash2 | Failed; deleted |
| Nov 8 – 10 | Restart: a lookup table built from the model's own text | 1.24× on mixed prompts (GPU pilot) |
| Nov 10 – 12 | Tables built from the model's own code, plus a reranker | 1.93× on coding prompts |
| Nov 15 – 17 | Wider GPU runs: HumanEval, relaxed acceptance, llama.cpp's drafters, the MTP head | `ngram-simple` in front: 3.3× on HumanEval |
| Nov 18 | The MTP head drafts the first token, MemDraft the rest | 2.16× coding, 3.60× HumanEval |
| Nov 18 – 19 | MemDraft becomes a llama.cpp plugin | Same speed as the patched build |
| Nov 15 | Side experiment: Qwen3.8 n-gram embeddings as reranker features | Built, never measured |

## 1. The first attempt: a continuation cache (failed)

**Idea.** Keep a table of token continuations seen earlier in the request stream, filled at run time. Put it in front of DFlash2, a strong learned drafter, so that repeated text could be drafted cheaply. Keys were exact 4- or 8-token sequences, drafts up to 16 tokens, and a small MLP decided when to use the cache.

**What happened.** On new requests the cache barely helped. With Qwen3.8-27B on CPU, tokens accepted per proposal went from 3.37 with DFlash2 alone to 3.39 to 3.53 with the cache. Acceptance fell, and the one timed comparison was 4% slower. It only paid off on exact repeats. On Qwen3.6-35B-A3B, every n-gram source placed in front of DFlash2 lowered tokens per step, MemDraft included. No clean timing was ever obtained, because every timing run got stuck in a queue on a shared machine.

**Why it failed.** Long drafts from a cache are usually wrong on new text, and each rejected token still costs verification time. DFlash2 was already good, and similar published work (Cacheback, SuffixDecoding, ReSpec) already covers the idea. The code was deleted on 2026-11-08.

**What carried over:**

- Speculative decoding in llama.cpp is not bit-exact. Greedy output depends on batch size, so speculative setups must be compared on task quality, not on identical text.
- Acceptance rate is not speed.
- A speedup on a tiny model means nothing.
- Measure on a quiet machine, as one job, planned from the start.

## 2. The restart: a table built ahead of time (Nov 8–10)

**New question.** Can a drafter that runs no model at all beat plain decoding in tokens per second, standing alone?

**Why it might.** A decode step with `n` draft tokens costs about `1 + β·n` plain steps, where β is the cost of one more token in the batch. On the 9B through Vulkan, β is about 0.065. When drafting is free, a guess is worth sending whenever its chance of being right is above about β. A guess right only one time in five still pays.

**The drafter.** For each of the last 1 to 6 tokens of context, a table counted from text the model wrote stores its most likely next token and a calibrated confidence. A second lookup does the same over the current request. A draft step takes about 2 µs.

**Offline study** (`tools/ngram_study.py`; results in `research/results/study-9b-*.json`). This used 35M tokens the 9B wrote for math and chat prompts, with a held-out test set:

- With β = 0.10, predicted speed was 1.38× using both lookups. Neither lookup alone did as well, because the table covers chat and the request covers math.
- The model's plain output is enough to build the table; no extra model pass is needed.
- Contexts seen fewer than three times can be dropped: 64 MiB loses almost nothing against 512 MiB.
- A table carries poorly outside the kind of text it was counted from.

**Two hidden costs in llama.cpp,** both found here and both hitting short drafts hardest:

1. **Rollback.** Qwen3.5 has recurrent layers. With only n-gram drafters enabled, llama.cpp saved a checkpoint before each drafted step and, after any rejection, restored it and decoded again. The patch asks for per-token snapshots instead, as model drafters get.
2. **Graph rebuilds.** llama.cpp rebuilds its compute graph whenever the batch size changes, which costs 5.7 ms on the 9B, as much as three draft tokens. Variable-length drafts change it at most steps. The fix is `pad`: always draft the same number of tokens. This one change turned a slowdown on chat into a gain.

**First live numbers** (pilot, busy machine):

- On the GPU with mixed prompts, two-token drafts gave 1.24×: 1.09× on chat, 1.18× on code, 1.32× on editing and 1.41× on math.
- On CPU, MemDraft gave 1.18× and llama.cpp's `ngram-cache` 0.99×.
- The β cost model predicted these within 4%.

## 3. Code, and the model's own code (Nov 10–12)

The first table had no code in it. Coding prompts turned out to suit lookup drafting well anyway, because code repeats its own names: the request lookup alone got 37% of first guesses right on code, against 13% on chat.

**The model's own code is worth far more than anyone else's.** The 9B answered 800 OSS-Instruct prompts. Offline:

- 0.22M tokens of its own code did better than the 19.5M-token general table.
- gpt-3.5's solutions to twice as many prompts did worse than the 9B's own code.

On CPU, a table with code in it gave 1.37×, against 1.21× for the general table.

**Scaling up.** The 9B then answered every remaining OSS-Instruct prompt: 4,200 on the local CPU and 69,000 on one rented A40 (37M tokens in 14 hours, $8.71). Each doubling of own code in the table still added about 0.02× offline at 39M tokens. The end of that curve has not been reached.

**A reranker.** Instead of taking the most confident proposal, a small MLP scores each proposed token using the lookups' statistics and the target's token embeddings. It added about 6% tokens per step offline. It was trained in [`research/reranker/`](../research/reranker/), and its C++ version matches the Python one on 99.975% of positions.

**First GPU result on code** (12 prompts, one round): 1.81× with the bigger table, and **1.93×** with the reranker drafting three tokens. On the GPU a drafted step costs about what a plain step costs, so speed tracks tokens per step almost one for one.

## 4. Wider GPU runs (Nov 15–17)

With GPU turns now taken only when the GPU was idle and the machine quiet, the runs became cleaner and larger.

**HumanEval.** On all 164 problems the table and reranker gave **2.29×** (coding prompts: 1.80×).

**Relaxed acceptance was not worth it.** A patch option kept a drafted token whenever the model gave it at least R times the probability of its own top choice. Lowering R from 1.0 to 0.1:

- did not speed up HumanEval (2.29× to 2.27×);
- lowered pass@1 from 83.5% to 80.5% (plain decoding: 84.8%).

MemDraft stays lossless.

**Against llama.cpp's own lookup drafters**, run as shipped (36 HumanEval problems, 12 coding prompts):

| Drafter | Coding prompts | HumanEval | Whole-file edits |
|---|---:|---:|---:|
| `ngram-cache` | 0.98× | 1.31× | 1.79× |
| `ngram-mod` | 1.07× | 1.72× | 2.39× |
| `ngram-simple` | 1.14× | 1.95× | 2.82× |
| MemDraft, 3 tokens | 1.81× | 2.39× | 2.48× |
| **`ngram-simple`, then MemDraft** | **1.83×** | **3.27×** | **3.85×** |

The two kinds of drafter complement each other. `ngram-simple` drafts long runs when the answer copies from the prompt, and MemDraft covers everything else. Chaining them became the default. Longer MemDraft drafts did not help: 5 tokens were about even with 3, and 8 tokens were slower than plain decoding on code.

**The model's own MTP head.** Qwen3.5 ships a multi-token-prediction head. Run alone through llama.cpp's `draft-mtp`, it gave 1.66× on code and 1.93× on HumanEval, both below MemDraft. It accepts more tokens per step, but each step costs 1.9 plain steps because the head is a real model.

## 5. The head drafts first (Nov 18)

The head is good at the next token, and MemDraft is cheap for the tokens after it. So the head now drafts one token and MemDraft continues from there. The head's output projection (248k × 4096) is most of its cost, so smaller quantizations were tried:

| Head | Coding prompts | HumanEval |
|---|---:|---:|
| none (`ngram-simple`, then MemDraft) | 1.84× | 3.15× |
| Q8_0 head, then MemDraft | 2.02× | 3.38× |
| Q4_K_M head with Q4_K output (572 MB), then MemDraft | **2.16×** | **3.60×** |

The cheapest head did best. This is the setup in the README.

## 6. A plugin instead of a patch (Nov 18–19)

Until now MemDraft lived inside a patched llama.cpp ([`research/patched-build/`](../research/patched-build/)). The patch was tied to one commit and touched llama.cpp's drafter code. The replacement is a small hook ([`patches/llama-plugin-hook.patch`](../patches/llama-plugin-hook.patch)): `--spec-type plugin` loads any drafter from a shared library, and `--spec-plugin-prefix` lets a draft model such as the MTP head go first.

**Equivalence.** MemDraft as a plugin ran at the same speed as the patched build: 1.85× against 1.84× on coding prompts, and 2.14× against 2.15× with the head.

**Example plugins.** To check that the hook is general, three of llama.cpp's built-in drafters were rewritten as plugins: `ngram-mod`, a separate draft model and the MTP head.

- The `ngram-mod` plugin drafts the same tokens as the built-in one and ran at the same speed.
- The draft-model plugin was slightly faster than the built-in.
- The MTP plugin was 13% slower at first. The cause was taking the argmax over the full vocabulary (0.68 ms per call). A top-10 selection (0.04 ms) brought it within 3%.

## Side experiment: Qwen3.8 n-gram embeddings (Nov 15, unfinished)

Qwen3.8-Flash-Next keeps learned embeddings for hashed n-grams. The idea was to feed those rows to the reranker as extra features. The feature extraction, the training code, a native scorer and head-selection tools are all built and tested ([`research/ple/`](../research/ple/), [`research/qwen38-head-selection/`](../research/qwen38-head-selection/)). The experiment was never run, because the machine did not have the model's weights or enough free memory.

## Still open

- A clean run on a quiet machine, with more prompts.
- Other models. The table should work for any model with the Qwen3.5 tokenizer (the 27B, 35B-A3B and 2B), but this has not been measured.
- Chat. The shipped table is built for code, and chat was the weakest domain in the general study.
- Where the own-code curve flattens, using prompts beyond OSS-Instruct.
- Publishing the table and reranker files.
