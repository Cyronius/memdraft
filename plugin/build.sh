#!/bin/bash
# Build libmemdraft.so, the MemDraft drafter for llama.cpp's --spec-type plugin.
#   ./plugin/build.sh LLAMA_CPP_DIR [OUT]
# LLAMA_CPP_DIR is a llama.cpp tree with common/speculative-plugin.h (llama-plugin-hook.patch applied).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
LLAMA=${1:?usage: build.sh LLAMA_CPP_DIR [OUT]}
OUT=${2:-$HERE/libmemdraft.so}
${CXX:-g++} -O2 -std=c++17 -shared -fPIC -fvisibility=hidden -Wall -Wextra \
  -I "$HERE/.." -I "$LLAMA/common" "$HERE/memdraft_plugin.cpp" -o "$OUT"
echo "built $OUT"
