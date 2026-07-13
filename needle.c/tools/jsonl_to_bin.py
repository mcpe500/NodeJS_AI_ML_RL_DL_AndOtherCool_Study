#!/usr/bin/env python3
"""JSONL (query/tools/call) → NDSET001 stream bin using tokenizer/vocab+merges."""
from __future__ import annotations
import argparse
import json
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_bpe(vocab_path, merges_path):
    tokens = {}
    inv = {}
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            tok_s, id_s = line.rsplit("\t", 1)
            tok = (
                tok_s.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\r")
                .replace("\\s", " ")
                .replace("\\\\", "\\")
            )
            i = int(id_s)
            tokens[tok] = i
            inv[i] = tok
    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            def un(s):
                return (
                    s.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\r", "\r")
                    .replace("\\s", " ")
                    .replace("\\\\", "\\")
                )
            merges.append((un(a), un(b)))
    specials = {
        "pad": tokens.get("<pad>", 0),
        "bos": tokens.get("<bos>", 1),
        "eos": tokens.get("<eos>", 2),
        "unk": tokens.get("<unk>", 3),
    }
    return tokens, inv, merges, specials


def encode(text, tokens, merges, specials, add_specials=True, max_len=128):
    toks = [c for c in text]
    for a, b in merges:
        i = 0
        while i + 1 < len(toks):
            if toks[i] == a and toks[i + 1] == b:
                toks[i] = a + b
                del toks[i + 1]
            else:
                i += 1
    ids = []
    if add_specials:
        ids.append(specials["bos"])
    for t in toks:
        ids.append(tokens.get(t, specials["unk"]))
    if add_specials:
        ids.append(specials["eos"])
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [specials["eos"]]
    return ids


def row_to_ids(row, tokens, merges, specials, max_src, max_tgt):
    src_text = f"Query: {row['query']}\nTools: {json.dumps(row['tools'], ensure_ascii=False, separators=(',', ':'))}"
    tgt_text = json.dumps(row["call"], ensure_ascii=False, separators=(',', ':'))
    src = encode(src_text, tokens, merges, specials, True, max_src)
    full = encode(tgt_text, tokens, merges, specials, True, max_tgt + 1)
    # teacher force: tgt = full[:-1], labels = full[1:]
    if len(full) < 2:
        full = [specials["bos"], specials["eos"]]
    tgt = full[:-1][:max_tgt]
    lab = full[1 : 1 + len(tgt)]
    while len(lab) < len(tgt):
        lab.append(specials["pad"])
    return src, tgt, lab


def write_ndset(path, rows, tokens, merges, specials, max_src, max_tgt):
    n = len(rows)
    samples = []
    for r in rows:
        src, tgt, lab = row_to_ids(r, tokens, merges, specials, max_src, max_tgt)
        samples.append((src, tgt, lab))

    with open(path, "wb") as f:
        f.write(b"NDSET001")
        f.write(struct.pack("<IIII", 1, n, max_src, max_tgt))
        off_table = f.tell()
        f.write(b"\x00" * (8 * n))
        offsets = []
        for src, tgt, lab in samples:
            offsets.append(f.tell())
            sl, tl = len(src), len(tgt)
            f.write(struct.pack("<II", sl, tl))
            f.write(struct.pack(f"<{sl}i", *src) if sl else b"")
            f.write(struct.pack(f"<{tl}i", *tgt) if tl else b"")
            f.write(struct.pack(f"<{tl}i", *lab) if tl else b"")
        end = f.tell()
        f.seek(off_table)
        for o in offsets:
            f.write(struct.pack("<Q", o))
        f.seek(end)
    print(f"wrote {path} n={n} max_src={max_src} max_tgt={max_tgt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=str(ROOT / "tokenizer/vocab.json"))
    ap.add_argument("--merges", default=str(ROOT / "tokenizer/merges.txt"))
    ap.add_argument("--max-src", type=int, default=128)
    ap.add_argument("--max-tgt", type=int, default=64)
    args = ap.parse_args()
    tokens, inv, merges, specials = load_bpe(args.vocab, args.merges)
    rows = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    write_ndset(args.out, rows, tokens, merges, specials, args.max_src, args.max_tgt)


if __name__ == "__main__":
    main()
