#!/usr/bin/env python3
"""
FULL version: clean, filter and split a plain-text file of lines/sentences (web-scrape style).

Input: one or more .txt files (or folders of .txt files), one sentence or paragraph per line.
Blank lines are treated as document boundaries if your file has them.

Per-line cleaning:
  - HTML entities and tags, control characters, NFC, unified quotes/apostrophes/dashes
  - emails, URLs, bullet markers, very long words
  - non-Latin scripts (Arabic, Greek, Cyrillic, ...) unless --keep-non-latin
  - sentences glued together after a full stop ("duniani.Iliundwa")
  - optional lowercasing (--lowercase; off by default so acronyms keep their capitals)

Line filters (each counted in report.json):
  too short, too many digits, too few letters, absurdly long, and lines that look English
  instead of Swahili (a crude function-word check; use --no-lang-filter to switch off)

Corpus-level steps:
  - boilerplate: lines that occur --max-line-repeat times or more are dropped entirely
  - exact duplicate lines are removed (first copy kept)
  - split by BLOCKS of neighbouring lines (no article ids exist), balanced by word count

Outputs: train/val/test .txt, train_wordcounts.txt (for Morfessor, from TRAIN only), report.json

Usage:
  python clean_lines_full.py data.txt --out out
  python clean_lines_full.py data/ --out out --block-size 30 --lowercase
"""

import argparse
import hashlib
import html
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
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.I)
BULLET_RE = re.compile(r"^\s*(?:[-*\u2022\u00b7\u25aa\u25ba>]+\s+|\d{1,3}[.)]\s+)")
NON_LATIN_RE = re.compile(
    "[\u0370-\u03ff\u1f00-\u1fff"
    "\u0400-\u052f\u0530-\u058f\u0590-\u05ff"
    "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"
    "\u0900-\u0dff\u0e00-\u0eff\u1000-\u109f\u10a0-\u10ff\u1200-\u137f"
    "\u1100-\u11ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+"
)
OPEN_PAREN_JUNK_RE = re.compile(r"\(\s*[,;:]+\s*")
CLOSE_PAREN_JUNK_RE = re.compile(r"[,;:]+\s*\)")
EMPTY_PARENS_RE = re.compile(r"\(\s*[,;:.\-\s]*\)")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
PUNCT_GLUE_RE = re.compile(r"(?<=[a-z])([.!?;])(?=[A-Z][a-z])")
EDGE_PUNCT = ".,;:!?()[]{}\"'*\u2026"

ENGLISH_STOP = set("""
the of and to in is was for with from by on as at are that this it or be which his her their has have
had not were also an who but they been its after first new one two into than other more most some such
only during between about over these those when where while will would could should can may might
you your we our he she them there here what how why all any each if then so do does did just very
""".split())
SWAHILI_FUNCTION = set("""
na ya wa kwa ni la za katika kuwa hii huo hiyo hizi hao yake wake lake zake kama pia lakini au ambayo
ambao ambaye ambazo ambalo cha vya kwamba wakati sana bado huu hili hilo kila mimi wewe yeye sisi nyinyi
wao ndani juu chini baada kabla hadi tu ila wala kwenye pamoja bila ndiyo siyo si ili hata mara tena sasa
hapa huko hapo baadhi wengi mengi yote zote wote vyote kutoka kuhusu
""".split())


def clean_line(line, a, stats):
    t = html.unescape(line)
    t = unicodedata.normalize("NFC", t)
    t = CONTROL_RE.sub(" ", t).translate(CHAR_MAP)
    t = HTML_TAG_RE.sub(" ", t)
    t, n = EMAIL_RE.subn(" ", t)
    stats["emails_removed"] += n
    t, n = URL_RE.subn(" ", t)
    stats["urls_removed"] += n
    t = BULLET_RE.sub("", t)
    if not a.keep_non_latin:
        t = NON_LATIN_RE.sub(" ", t)
        t = OPEN_PAREN_JUNK_RE.sub("(", t)
        t = CLOSE_PAREN_JUNK_RE.sub(")", t)
        t = EMPTY_PARENS_RE.sub(" ", t)
    t = PUNCT_GLUE_RE.sub(r"\1 ", t)
    kept = []
    for tok in t.split():
        if len(tok.strip(EDGE_PUNCT)) > a.max_word_len:
            stats["long_words_removed"] += 1
            continue
        kept.append(tok)
    t = SPACE_BEFORE_PUNCT_RE.sub(r"\1", " ".join(kept))
    if a.lowercase:
        t = t.lower()
    else:
        t = t.lower()
    return unicodedata.normalize("NFC", t)

def reject_reason(line, a):
    """Return why a cleaned line should be dropped, or None to keep it."""
    n = len(line)
    if n == 0:
        return "empty"
    if n > a.max_line_chars:
        return "too_long"
    words = line.split()
    if len(words) < a.min_line_words:
        return "too_short"
    if sum(c.isdigit() for c in line) / n > a.max_digit_ratio:
        return "digit_heavy"
    if sum(c.isalpha() for c in line) / n < a.min_alpha_ratio:
        return "low_letter_share"
    if not a.no_lang_filter:
        toks = re.findall(r"[a-z']+", line.lower())
        if len(toks) >= 6:
            en = sum(t in ENGLISH_STOP for t in toks) / len(toks)
            sw = sum(t in SWAHILI_FUNCTION for t in toks) / len(toks)
            if en >= 0.25 and en > sw:
                return "looks_english"
    return None


def line_hash(line):
    return hashlib.blake2b(line.lower().encode("utf-8"), digest_size=8).digest()


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
    for b in blocks:
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


WORD_FULL_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
REPEAT_RE = re.compile(r"(.)\1{3,}")


def build_word_counts(lines, min_count, english):
    counts = Counter()
    for line in lines:
        for tok in line.lower().split():
            tok = tok.strip(EDGE_PUNCT + "-_/")
            if WORD_FULL_RE.fullmatch(tok):
                counts[tok] += 1
    kept = {}
    for w, c in counts.items():
        if c < min_count or not (2 <= len(w) <= 30) or w in english:
            continue
        if not re.search(r"[aeiou]", w) or re.search(r"[qx]", w) or REPEAT_RE.search(w):
            continue
        kept[w] = c
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="+", help="Text file(s) or folder(s) of .txt files")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--block-size", type=int, default=50, help="Max neighbouring lines kept together")
    ap.add_argument("--max-word-len", type=int, default=30)
    ap.add_argument("--min-line-words", type=int, default=3)
    ap.add_argument("--max-line-chars", type=int, default=3000)
    ap.add_argument("--max-digit-ratio", type=float, default=0.3)
    ap.add_argument("--min-alpha-ratio", type=float, default=0.6)
    ap.add_argument("--max-line-repeat", type=int, default=10,
                    help="Drop a line entirely if it occurs this many times or more (boilerplate)")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase the output (off by default)")
    ap.add_argument("--keep-non-latin", action="store_true")
    ap.add_argument("--no-lang-filter", action="store_true", help="Do not drop lines that look English")
    ap.add_argument("--min-word-count", type=int, default=2, help="Min frequency for the Morfessor word list")
    ap.add_argument("--english-words", help="Optional file with one English word per line to exclude")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats, dropped = Counter(), Counter()

    # 1. Read and clean -------------------------------------------------------
    raw_blocks = read_blocks(expand_paths(a.input), a.block_size)
    n_lines = sum(len(b) for b in raw_blocks)
    print(f"Read {n_lines:,} lines in {len(raw_blocks):,} blocks")
    if not raw_blocks:
        sys.exit("The input has no text.")
    blocks = []
    for b in raw_blocks:
        kept = []
        for line in b:
            c = clean_line(line, a, stats)
            why = reject_reason(c, a)
            if why:
                dropped[why] += 1
            else:
                kept.append(c)
        if kept:
            blocks.append(kept)

    # 2. Boilerplate: same line repeated many times anywhere in the data ------
    counts = Counter(line_hash(l) for b in blocks for l in b)
    boiler = {h for h, c in counts.items() if c >= a.max_line_repeat}
    if boiler:
        before = sum(len(b) for b in blocks)
        blocks = [[l for l in b if line_hash(l) not in boiler] for b in blocks]
        dropped["boilerplate_lines"] = before - sum(len(b) for b in blocks)

    # 3. Exact duplicates (first copy kept) -----------------------------------
    seen, deduped = set(), []
    for b in blocks:
        kept = []
        for l in b:
            h = line_hash(l)
            if h in seen:
                dropped["duplicate_lines"] += 1
                continue
            seen.add(h)
            kept.append(l)
        if kept:
            deduped.append(kept)
    blocks = deduped
    if len(blocks) < 3:
        sys.exit("Fewer than 3 blocks left. Lower --block-size or loosen the filters.")

    # 4. Split by block -------------------------------------------------------
    assign, words = split_blocks(blocks, a.val_frac, a.test_frac, a.seed)
    report = {"lines_read": n_lines, "cleaning": dict(stats), "dropped": dict(dropped), "splits": {}}
    for split in ("train", "val", "test"):
        idx = [i for i in range(len(blocks)) if assign[i] == split]
        with open(out / f"{split}.txt", "w", encoding="utf-8") as f:
            for i in idx:
                f.write("\n".join(blocks[i]) + "\n\n")
        report["splits"][split] = {"blocks": len(idx), "lines": sum(len(blocks[i]) for i in idx),
                                   "words": sum(words[i] for i in idx)}

    # 5. Morfessor word list from TRAIN only ----------------------------------
    english = set(ENGLISH_STOP)
    if a.english_words:
        english |= {w.strip().lower() for w in open(a.english_words, encoding="utf-8") if w.strip()}
    train_lines = (l for i, b in enumerate(blocks) if assign[i] == "train" for l in b)
    wc = build_word_counts(train_lines, a.min_word_count, english)
    with open(out / "train_wordcounts.txt", "w", encoding="utf-8") as f:
        for w, c in sorted(wc.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{c} {w}\n")
    report["morfessor_word_types"] = len(wc)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"  cleaning: {dict(stats)}")
    print(f"  dropped:  {dict(dropped)}")
    for split, s in report["splits"].items():
        print(f"  {split:<5} {s['blocks']:>6,} blocks {s['lines']:>9,} lines {s['words']:>11,} words")
    print(f"  Morfessor word types: {len(wc):,}")
    print(f"Files written to {out.resolve()}")


if __name__ == "__main__":
    main()
