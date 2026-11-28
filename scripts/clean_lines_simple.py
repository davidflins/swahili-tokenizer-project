#!/usr/bin/env python3
"""
SIMPLE version: clean a plain-text file of lines/sentences and split it into train / val / test.

Input: one or more .txt files (or folders of .txt files), one sentence or paragraph per line.
Blank lines are treated as document boundaries if your file has them.

Cleaning, in this order:
  1. Normalize to Unicode NFC.
  2. Unify curly/straight quotes, apostrophes and dashes.
  3. Drop emails, URLs and very long words.
  4. Lowercase everything.
Then exact-duplicate lines are removed (keeping the first copy).

Splitting: there are no article ids in this kind of data, so the split is done by BLOCKS of
neighbouring lines (a blank-line-separated block, cut into pieces of at most --block-size lines).
Neighbouring sentences usually come from the same page, so keeping them together stops
near-copies leaking between train and test. Splits are balanced by word count.

Usage:
  python clean_lines_simple.py data.txt --out out
  python clean_lines_simple.py data/ --out out --block-size 30 --val-frac 0.1 --test-frac 0.1
"""

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

CHAR_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u02bc": "'", "\u02bb": "'", "`": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u00a0": " ",
})
EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.I)
EDGE_PUNCT = ".,;:!?()[]{}\"'*\u2026"


def clean(line, max_word_len, stats):
    t = unicodedata.normalize("NFC", line).translate(CHAR_MAP)
    t, n = EMAIL_RE.subn(" ", t)
    stats["emails_removed"] += n
    t, n = URL_RE.subn(" ", t)
    stats["urls_removed"] += n
    kept = []
    for tok in t.split():
        if len(tok.strip(EDGE_PUNCT)) > max_word_len:
            stats["long_words_removed"] += 1
            continue
        kept.append(tok)
    return unicodedata.normalize("NFC", " ".join(kept).lower())


def expand_paths(inputs):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(p.rglob("*.txt"))
        elif p.exists():
            paths.append(p)
        else:
            sys.exit(f"Not found: {item}")
    if not paths:
        sys.exit("No input files found.")
    return paths


def read_blocks(paths, block_size):
    blocks, cur = [], []
    for p in paths:
        with open(p, encoding="utf-8-sig", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.strip():
                    cur.append(line)
                elif cur:
                    blocks.append(cur); cur = []
        if cur:
            blocks.append(cur); cur = []
    out = []
    for b in blocks:                                   # cut long blocks into consecutive chunks
        out += [b[i:i + block_size] for i in range(0, len(b), block_size)]
    return out


def split_blocks(blocks, val_frac, test_frac, seed):
    words = [sum(len(l.split()) for l in b) for b in blocks]
    total = sum(words)
    order = list(range(len(blocks)))
    random.Random(seed).shuffle(order)
    test_target, val_target = test_frac * total, val_frac * total
    assign, test_w, val_w = {}, 0, 0
    for i in order:
        if test_w < test_target:
            assign[i] = "test"; test_w += words[i]
        elif val_w < val_target:
            assign[i] = "val"; val_w += words[i]
        else:
            assign[i] = "train"
    return assign, words


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="+", help="Text file(s) or folder(s) of .txt files")
    ap.add_argument("--out", default="out", help="Output folder (created if missing)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.05, help="Validation share, by word count")
    ap.add_argument("--test-frac", type=float, default=0.05, help="Test share, by word count")
    ap.add_argument("--max-word-len", type=int, default=30, help="Words longer than this are dropped")
    ap.add_argument("--block-size", type=int, default=50,
                    help="Max neighbouring lines kept together when splitting")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw_blocks = read_blocks(expand_paths(args.input), args.block_size)
    n_lines = sum(len(b) for b in raw_blocks)
    print(f"Read {n_lines:,} lines in {len(raw_blocks):,} blocks")
    if not raw_blocks:
        sys.exit("The input has no text.")

    stats = Counter()
    seen, blocks = set(), []
    for b in raw_blocks:
        kept = []
        for line in b:
            c = clean(line, args.max_word_len, stats)
            if not c:
                stats["empty_after_cleaning"] += 1
                continue
            h = hashlib.md5(c.encode("utf-8")).digest()
            if h in seen:
                stats["duplicate_lines_removed"] += 1
                continue
            seen.add(h)
            kept.append(c)
        if kept:
            blocks.append(kept)
    if len(blocks) < 3:
        sys.exit("Fewer than 3 blocks left. Lower --block-size so there is something to split.")

    assign, words = split_blocks(blocks, args.val_frac, args.test_frac, args.seed)
    report = {"lines_read": n_lines, "cleaning": dict(stats), "splits": {}}
    for split in ("train", "val", "test"):
        idx = [i for i in range(len(blocks)) if assign[i] == split]
        with open(out / f"{split}.txt", "w", encoding="utf-8") as f:
            for i in idx:
                f.write("\n".join(blocks[i]) + "\n\n")
        report["splits"][split] = {"blocks": len(idx), "lines": sum(len(blocks[i]) for i in idx),
                                   "words": sum(words[i] for i in idx)}
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"  cleaning: {dict(stats)}")
    for split, s in report["splits"].items():
        print(f"  {split:<5} {s['blocks']:>6,} blocks {s['lines']:>9,} lines {s['words']:>11,} words")
    print(f"Files written to {out.resolve()}")


if __name__ == "__main__":
    main()
