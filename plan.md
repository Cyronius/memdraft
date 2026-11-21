# README follow-ups

To do on the machine that ran the measurements (it has `data/`, `$MEMDRAFT_CACHE` and the GPU). Written 2026-11-21 from a review of the README.

## Needs a GPU run

1. **Put the whole results table on one run.** "MemDraft alone" and "`ngram-simple` alone" come from `research/results/prior-9b-vk-20261117-1614/` (patched build, one round of coding prompts, HumanEval 0–35). The other rows come from `bench/results/plugin-9b-vk-20261118-2046/` (plugin build, two rounds, HumanEval 0–11), but the column header says 0–11 for all rows.
   - Add arms to `bench/run_plugin_live.sh`: `plugin-3` (`--spec-type plugin --spec-plugin-n-max 3 $P`) and `simple` (`--spec-type ngram-simple`).
   - Rerun with the current arms. Then replace the table and drop footnote ¹.
2. **Settle MTP head alone against MemDraft alone on coding prompts.** 1.94× (1.82–2.08) against 1.81× (1.66–1.96): the intervals overlap. Run a paired comparison over the same prompts, or more prompts (`code-v2.jsonl` has 160 held-out ones). Until then, soften the README bullet that says the head "beats" MemDraft.

## Doc-only

3. **Say up front that no ready-made table exists.** The intro says the table "is not in this repo", which reads as "it is somewhere else". State that you must build one to try MemDraft. Also say that the results table includes unpublished chat and math data, and that reranker training is research code.
4. **Show what a smaller table buys.** If `research/results/code-study*.json` has speedup or tokens-per-step against table size, add one line or a small table ("N M tokens → X×"). This tells a reader whether 40M tokens / 14 A40-hours is really needed.
5. **Show the 95% intervals.** `summary.txt` has them. Put them in the speedup cells, such as `1.85× (1.69–2.05)`, or in a column.
6. **Fix the Results layout.** The "two other drafters" list runs straight into the findings list with no lead-in. Add a lead-in, and move the "setup is narrow so far" caveat to the top of Results or into the intro.
7. **State the platform.** The plugin is a `.so` built by `plugin/build.sh`. Say Linux only (or whatever was tested).
8. **`pad` default.** The default is 0, but the docs say "use 1". Change the default to 1, or say why it stays 0.
9. **Quick start `vocab_hash`.** `c52114e74bc3a20f` is the Qwen3.5 tokenizer hash. Say so, or use a placeholder with the `vocab_hash.py` command next to it.
10. **Output divergence bullet.** Text matched plain decoding on only 8 of 24 coding runs. Lead with "plain decoding diverges the same way when only the batch size changes", then give the counts. If possible, measure plain decoding at two batch sizes and give that match count too.

When done, delete this file.
