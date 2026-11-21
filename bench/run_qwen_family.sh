#!/bin/bash
# The published Qwen3.5-9B drafter, unchanged, on the other Qwen3.5 sizes (2026-11-21).
#   setsid nohup bench/run_qwen_family.sh > data/code/family.log 2>&1 < /dev/null &
# Overrides: SIZES (default "0.8B 2B 4B 9B 27B 35B-A3B"), ARMS, and anything run_plugin_live.sh takes.
set -u
cd "$(dirname "$0")/.."
S=${MEMDRAFT_CACHE:-$HOME/.cache/model-experiments/memdraft}
ARMS=${ARMS:-simple plugin-3 copy+plugin-3 copy+plugin-3-norr}
for size in ${SIZES:-0.8B 2B 4B 9B 27B 35B-A3B}; do
  f=Qwen3.5-$size-Q4_K_M.gguf
  if [ ! -f "$S/models/$f" ]; then
    curl -fL --retry 3 -o "$S/models/$f.part" "https://huggingface.co/unsloth/Qwen3.5-$size-GGUF/resolve/main/$f" &&
      mv "$S/models/$f.part" "$S/models/$f" || { echo "$(date +%T) $size: download failed"; continue; }
  fi
  tag=$(echo "$size" | tr 'A-Z.' 'a-z_')
  echo "$(date +%T) $size: start"
  MODEL=$S/models/$f TAG=$tag HEAD= PATCHED_ARMS= PLUGIN_ARMS="$ARMS" bench/run_plugin_live.sh
  echo "$(date +%T) $size: exit $?"
done
