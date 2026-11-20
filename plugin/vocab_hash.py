#!/usr/bin/env python3
"""Print the vocabulary hash llama.cpp passes to a drafter plugin, for a GGUF model.

FNV-1a 64 over the text of every token in tokenizer.ggml.tokens, each followed by a 0 byte, the same
as common_speculative_impl_plugin computes from llama_vocab_get_text. Give it to MemDraft as
--spec-plugin-args ...,vocab_hash=HEX so a table built for another tokenizer refuses to load.

    python vocab_hash.py MODEL.gguf [GGUF_PY_DIR]
"""
import sys

if len(sys.argv) > 2:
    sys.path.insert(0, sys.argv[2])
import gguf  # noqa: E402

r = gguf.GGUFReader(sys.argv[1])
f = r.fields['tokenizer.ggml.tokens']
h = 0xcbf29ce484222325
for i in f.data:
    for b in bytes(f.parts[i]) + b'\0':
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
print(f'{h:016x}')
