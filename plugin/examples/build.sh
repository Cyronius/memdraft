#!/bin/bash
# Build the example plugins and the equivalence test against a llama.cpp tree with the plugin hook.
#   ./plugin/examples/build.sh LLAMA_CPP_DIR [BUILD_DIR]     (BUILD_DIR default: LLAMA_CPP_DIR/build-vk)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
LLAMA=${1:?usage: build.sh LLAMA_CPP_DIR [BUILD_DIR]}
BIN=${2:-$LLAMA/build-vk}/bin
INC=(-I "$LLAMA/common" -I "$LLAMA/include" -I "$LLAMA/ggml/include" -I "$LLAMA/vendor")
LINK=(-L "$BIN" -Wl,-rpath,"$BIN")
CXX=${CXX:-g++}
$CXX -O2 -std=c++17 -shared -fPIC -fvisibility=hidden -Wall "${INC[@]}" "$HERE/builtin_ngram_plugin.cpp" \
  -o "$HERE/libbuiltin_ngram.so" "${LINK[@]}" -lllama-common -lllama
$CXX -O2 -std=c++17 -shared -fPIC -fvisibility=hidden -Wall "${INC[@]}" "$HERE/draft_model_plugin.cpp" \
  -o "$HERE/libdraft_model.so" "${LINK[@]}" -lllama
$CXX -O2 -std=c++17 -shared -fPIC -fvisibility=hidden -Wall "${INC[@]}" "$HERE/mtp_plugin.cpp" \
  -o "$HERE/libmtp.so" "${LINK[@]}" -lllama
$CXX -O2 -std=c++17 -Wall "${INC[@]}" "$HERE/test_plugin_equiv.cpp" -o "$HERE/test_plugin_equiv" "${LINK[@]}" -lllama-common -lllama
$CXX -O2 -std=c++17 -Wall "${INC[@]}" "$HERE/test_plugin_replay.cpp" -o "$HERE/test_plugin_replay" "${LINK[@]}" -lllama-common -lllama
echo "built libbuiltin_ngram.so libdraft_model.so libmtp.so test_plugin_equiv test_plugin_replay in $HERE"
