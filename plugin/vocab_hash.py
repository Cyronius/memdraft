#!/usr/bin/env python3
"""Print the vocabulary hash llama.cpp passes to a drafter plugin, for a GGUF model.

FNV-1a 64 over the text of every token in tokenizer.ggml.tokens, each followed by a 0 byte, the same
as common_speculative_impl_plugin computes from llama_vocab_get_text. Give it to MemDraft as
--spec-plugin-args ...,vocab_hash=HEX so a table built for another tokenizer refuses to load.

    python vocab_hash.py MODEL.gguf

Reads the GGUF header itself, so it needs no packages.
"""
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def u32(f):
    return struct.unpack('<I', f.read(4))[0]


def u64(f):
    return struct.unpack('<Q', f.read(8))[0]


def string(f):
    return f.read(u64(f))


def skip(f, t):
    if t in SIZES:
        f.seek(SIZES[t], 1)
    elif t == 8:
        f.seek(u64(f), 1)
    elif t == 9:
        et, n = u32(f), u64(f)
        if et in SIZES:
            f.seek(SIZES[et] * n, 1)
        else:
            for _ in range(n):
                skip(f, et)
    else:
        raise SystemExit(f'unknown GGUF value type {t}')


with open(sys.argv[1], 'rb') as f:
    if f.read(4) != b'GGUF':
        raise SystemExit(f'{sys.argv[1]} is not a GGUF file')
    u32(f), u64(f)
    for _ in range(u64(f)):
        key, t = string(f), u32(f)
        if key != b'tokenizer.ggml.tokens':
            skip(f, t)
            continue
        if t != 9 or u32(f) != 8:
            raise SystemExit('tokenizer.ggml.tokens is not an array of strings')
        h = 0xcbf29ce484222325
        for _ in range(u64(f)):
            for b in string(f) + b'\0':
                h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        print(f'{h:016x}')
        break
    else:
        raise SystemExit('no tokenizer.ggml.tokens in this file')
