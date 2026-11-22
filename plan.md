# README follow-ups

To do on the machine that ran the measurements (it has `data/`, `$MEMDRAFT_CACHE` and the GPU). Written 2026-11-21 from a review of the README.

## Needs a GPU run

2. **Settle MTP head alone against MemDraft alone on coding prompts.** 73.9 tok/s (1.94×, 95% interval 1.82–2.08) against 68.7 tok/s (1.81×, 1.66–1.96): the intervals overlap. Run a paired comparison over the same prompts, or more prompts (`code-v2.jsonl` has 160 held-out ones). Until then, soften the README bullet that says the head "beats" MemDraft.

## Doc-only

3. **Show what a smaller table buys.** If `research/results/code-study*.json` has speedup or tokens-per-step against table size, add one line or a small table ("N M tokens → T tok/s (X×)"). This tells a reader whether 40M tokens / 14 A40-hours is really needed.
4. **Show the 95% intervals.** `summary.txt` has them. Put them inside the parentheses, such as `70.5 tok/s (1.85×, 1.69–2.05)`, or in a column.
6. **State the platform.** The plugin is a `.so` built by `plugin/build.sh`. Say Linux only (or whatever was tested).
7. **`pad` default.** The default is 0, but the docs say "use 1". Change the default to 1, or say why it stays 0.
8. **Quick start `vocab_hash`.** `c52114e74bc3a20f` is the Qwen3.5 tokenizer hash. Say so, or use a placeholder with the `vocab_hash.py` command next to it.
9. **Output divergence bullet.** Text matched plain decoding on only 8 of 24 coding runs. Lead with "plain decoding diverges the same way when only the batch size changes", then give the counts. If possible, measure plain decoding at two batch sizes and give that match count too.

When done, delete this file.
