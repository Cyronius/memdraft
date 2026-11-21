# Research

This folder holds the experiments behind MemDraft. [docs/history.md](../docs/history.md) tells the story; this page says where each piece lives. The code is kept as it was run, so expect research-grade scripts. Run everything from the repo root. The shared Python modules are in [`tools/`](../tools/).

| Folder | What it is |
|---|---|
| `experiments/` | The live benchmark scripts behind each step of the history. Each script's header lists its arms and prompts. |
| `results/` | Raw results from those runs: per-turn JSON, server logs and `summary.txt`. Also the offline studies (`study-9b-*.json`, `code-study*.json`). |
| `code-generation/` | Having the 9B answer coding prompts, locally (`run_gen_code.sh`, `run_gen_more.sh`) and on a rented GPU (`run_gen_pod.py` and its worker). Also `make_prompts.py`, which built the prompt sets in `bench/prompts/`. |
| `reranker/` | Reranker training, `rerank.py` to `rerank8.py` in the order they were written. `rerank7.py` trained the reranker used in the README results, and `tools/export_reranker.py` exports it. |
| `patched-build/` | `patch_llama.py`, the earlier integration that patched MemDraft directly into llama.cpp `7d7f028`. It also has options that never shipped: relaxed acceptance (`--spec-accept-ratio`), head-first drafting and the PLE scorer. |
| `ple/` | The Qwen3.8 n-gram-embedding reranker experiment. Built and tested (`python -m pytest research/ple`), but never measured. |
| `qwen38-head-selection/` | Head calibration and GGUF surgery for the PLE experiment |

Some scripts need things that are not in this repo:

- `run_gen_pod.py` imports a pod launcher from a separate, unpublished checkout (`$MODEL_EXPERIMENTS`).
- The general-text studies read 9B output shards from that same checkout.

Logs from before the move to this repo may mention `memdraft/` or `target/memdraft/`. Today those are the repo root and `data/`.
