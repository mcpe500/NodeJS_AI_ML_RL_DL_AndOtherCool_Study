#!/usr/bin/env python3
"""Train minimal BPE for needle.c. Writes vocab.json (token\\tid lines) + merges.txt.
Vocab: 512 smoke default, 8192 target via --vocab-size.
"""
from __future__ import annotations
import argparse
import collections
import os
import pathlib
import re
import sys

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Seed corpus: JSON punctuation + tool-call phrases (ASCII)
SEED = [
    'Query: What\'s the weather in San Francisco?\nTools: [{"name":"get_weather","parameters":{"location":"string"}}]',
    '[{"name":"get_weather","arguments":{"location":"San Francisco"}}]',
    'Query: set an alarm for 6:30\nTools: [{"name":"create_alarm","parameters":{"time":"string"}}]',
    '[{"name":"create_alarm","arguments":{"time":"06:30"}}]',
    'Query: hello\nTools: []',
    '[]',
    'Query: search for needle papers\nTools: [{"name":"web_search","parameters":{"q":"string"}}]',
    '[{"name":"web_search","arguments":{"q":"needle papers"}}]',
    'Query: turn on the lights in the kitchen\nTools: [{"name":"set_light","parameters":{"room":"string","on":"boolean"}}]',
    '[{"name":"set_light","arguments":{"room":"kitchen","on":true}}]',
    '{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}',
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    '{}[]():,"\'\\/_-+=.!? \n\t',
    'Query: apa cuaca di Jakarta?\nTools: [{"name":"get_weather","parameters":{"location":"string"}}]',
    '[{"name":"get_weather","arguments":{"location":"Jakarta"}}]',
]


def escape_tok(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace(" ", "\\s")
    )


def train(corpus: list[str], vocab_size: int):
    # byte/char vocab
    chars = set()
    for t in corpus:
        chars.update(t)
    # base tokens = specials + sorted chars
    base = list(SPECIALS)
    for c in sorted(chars):
        if c not in base:
            base.append(c)
    # word = list of chars
    words = []
    for t in corpus:
        words.append([c for c in t])

    merges = []
    vocab = {tok: i for i, tok in enumerate(base)}

    def pair_stats(ws):
        st = collections.Counter()
        for w in ws:
            for i in range(len(w) - 1):
                st[(w[i], w[i + 1])] += 1
        return st

    def merge_pair(ws, a, b, ab):
        out = []
        for w in ws:
            nw, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    nw.append(ab)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            out.append(nw)
        return out

    # Cap merge length so encode stays multi-token (giant phrase tokens break learning).
    max_tok_len = 16
    while len(vocab) < vocab_size:
        st = pair_stats(words)
        if not st:
            break
        # pick most common pair whose merge stays ≤ max_tok_len and not already in vocab
        picked = None
        for (a, b), _cnt in st.most_common():
            ab = a + b
            if len(ab) > max_tok_len:
                continue
            if ab in vocab:
                # already known — still apply to collapse corpus, no new id
                words = merge_pair(words, a, b, ab)
                merges.append((a, b))
                picked = "applied"
                break
            picked = (a, b, ab)
            break
        if picked is None:
            break  # only oversize pairs left
        if picked == "applied":
            continue
        a, b, ab = picked
        vocab[ab] = len(vocab)
        merges.append((a, b))
        words = merge_pair(words, a, b, ab)

    return vocab, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).resolve().parent))
    ap.add_argument("--extra-corpus", default="", help="optional text file, one doc per line")
    args = ap.parse_args()
    corpus = list(SEED)
    if args.extra_corpus and os.path.isfile(args.extra_corpus):
        with open(args.extra_corpus, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    corpus.append(line)
    vocab, merges = train(corpus, args.vocab_size)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # vocab.json: token\tid lines (simple, not full JSON — name kept for plan)
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        # write in id order
        inv = [None] * len(vocab)
        for t, i in vocab.items():
            if 0 <= i < len(inv):
                inv[i] = t
        for i, t in enumerate(inv):
            if t is None:
                t = f"<unused_{i}>"
            f.write(f"{escape_tok(t)}\t{i}\n")
    with open(out / "merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{escape_tok(a)} {escape_tok(b)}\n")
    print(f"wrote {out}/vocab.json size={len(vocab)} merges={len(merges)}")


if __name__ == "__main__":
    main()
