#!/usr/bin/env python3
"""
Clean the Swahili Wikipedia dataset and split it into train / validation / test.

What it does, in order:
  1. Loads only the columns it needs (text, title, article_url).
  2. Cleans each article: wiki markup, tables, image maps, URLs, categories,
     interlanguage links, non-Latin scripts, odd quotes/apostrophes, glued sentences.
  3. Drops boilerplate lines that repeat across many articles.
  4. Drops junk articles (too short, too many digits, too few letters).
  5. Removes duplicate articles (same text or same title).
  6. Splits by whole article (never by sentence), balanced by word count.
  7. Removes lines from validation/test that also appear in train (leakage guard).
  8. Writes the splits, plain-text files for BPE training, and a word-count
     list for Morfessor built from the TRAIN split only.

Usage examples:
  python clean_swahili_wiki.py --input ffatty/plaintext-wikipedia-full-swahili --out out
  python clean_swahili_wiki.py --input data/train-00000.parquet --out out
  python clean_swahili_wiki.py --input data/ --out out --limit 2000     # quick test run

Requires: pandas, pyarrow. Also `datasets` if you load straight from the Hugging Face Hub.
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

import pandas as pd

NEEDED_COLUMNS = ["text", "title", "article_url"]

# --------------------------------------------------------------------------
# Regular expressions
# --------------------------------------------------------------------------
CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060\ufeff]"
)
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)               # {| ... |} wiki tables
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")                # {{template}}
FILE_LINK_RE = re.compile(r"\[\[(?:Image|File|Picha|Faili)\s*:[^\]]*\]\]", re.I)
WIKILINK_RE = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]")   # [[a|b]] -> b
EXT_LINK_RE = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")     # [http://x label] -> label
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)      # also catches URLs glued to words
HEADING_RE = re.compile(r"={2,}[^=\n]{1,100}={2,}")        # == Heading ==
CATEGORY_RE = re.compile(r"Jamii\s*:[^\n]*")               # category lines (to end of line)

# Lines that are markup/junk rather than prose. Checked before token-level cleanup.
BAD_LINE_RE = re.compile(
    r"^(?:image|file|picha|faili)\s*:"                     # image lines
    r"|^(?:poly|rect|circle)\s"                            # image-map coordinates
    r"|^[a-z]{2,3}(?:-[a-z]{2,8})?:\S"                     # interlanguage link (nl:...)
    r"|\b(?:style|class|width|height|colspan|rowspan|cellpadding|cellspacing"
    r"|bgcolor|valign|align)\s*="                          # html/css attributes
    r"|\b(?:fontsize|font-size)\b|\brgb\("                 # chart/style leftovers
    r"|^\s*\|",                                            # table rows
    re.I,
)
BULLET_RE = re.compile(r"^[*#:;]+\s*")
INTERWIKI_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[a-z]{2,3}(?:-[a-z]{2,8})?:[^\s:]\S*")
IMG_TOKEN_RE = re.compile(r"\S+\.(?:jpe?g|png|gif|svg|tiff?|webp|ogg|mp3|pdf)\b", re.I)
THUMB_RE = re.compile(r"\b(?:thumb|thumbnail)\b|\b\d+px\b", re.I)

# Characters from scripts other than Latin (Arabic, Greek, Hebrew, Cyrillic, Indic, CJK, ...)
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
PUNCT_GLUE_RE = re.compile(r"(?<=[a-z])([.!?;])(?=[A-Z][a-z])")   # "duniani.Iliundwa"
MULTISPACE_RE = re.compile(r"[ \t]+")

# Normalise look-alike characters so the tokenizer doesn't see them as different symbols.
CHAR_MAP = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u02bb": "'", "`": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
})

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_text(text, strip_non_latin=True, min_line_words=3, max_line_digit=0.2, min_line_alpha=0.6):
    """Return the cleaned article as a list of lines (paragraphs)."""
    t = html.unescape(text or "")
    t = unicodedata.normalize("NFC", t)
    t = CONTROL_RE.sub(" ", t)

    # Block-level markup
    t = TABLE_RE.sub("\n", t)
    for _ in range(3):                       # nested templates
        t = TEMPLATE_RE.sub(" ", t)
    t = FILE_LINK_RE.sub(" ", t)
    t = EXT_LINK_RE.sub(r"\1", t)
    t = WIKILINK_RE.sub(r"\1", t)
    t = HTML_TAG_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = HEADING_RE.sub("\n", t)
    t = CATEGORY_RE.sub("", t)
    t = t.translate(CHAR_MAP)

    lines = []
    for raw in t.split("\n"):
        line = raw.strip()
        if not line or BAD_LINE_RE.search(line):
            continue
        line = BULLET_RE.sub("", line)

        # Token-level cleanup
        line = INTERWIKI_TOKEN_RE.sub(" ", line)
        line = IMG_TOKEN_RE.sub(" ", line)
        line = THUMB_RE.sub(" ", line)
        if strip_non_latin:
            line = NON_LATIN_RE.sub(" ", line)
            line = OPEN_PAREN_JUNK_RE.sub("(", line)
            line = CLOSE_PAREN_JUNK_RE.sub(")", line)
            line = EMPTY_PARENS_RE.sub(" ", line)
        line = PUNCT_GLUE_RE.sub(r"\1 ", line)
        line = MULTISPACE_RE.sub(" ", line).strip()
        line = SPACE_BEFORE_PUNCT_RE.sub(r"\1", line)

        if line_ok(line, min_line_words, max_line_digit, min_line_alpha):
            lines.append(line)
    return lines


def line_ok(line, min_words, max_digit, min_alpha):
    n = len(line)
    if n == 0:
        return False
    letters = sum(ch.isalpha() for ch in line)
    digits = sum(ch.isdigit() for ch in line)
    if letters / n < min_alpha or digits / n > max_digit:
        return False
    return len(line.split()) >= min_words


def line_hash(line):
    return hashlib.blake2b(line.lower().encode("utf-8"), digest_size=8).digest()


def article_ratios(lines):
    text = "\n".join(lines)
    n = max(len(text), 1)
    return {
        "n_words": len(text.split()),
        "digit_ratio": sum(c.isdigit() for c in text) / n,
        "alpha_ratio": sum(c.isalpha() for c in text) / n,
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_articles(source, limit=None):
    path = Path(source)
    if path.exists():
        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        if not files:
            sys.exit(f"No .parquet files found under {source}")
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f, columns=NEEDED_COLUMNS))
            except Exception:
                # article_url or title may be missing; fall back to whatever exists
                df_ = pd.read_parquet(f)
                frames.append(df_[[c for c in NEEDED_COLUMNS if c in df_.columns]])
            if limit and sum(len(x) for x in frames) >= limit:
                break
        df = pd.concat(frames, ignore_index=True)
    else:
        from datasets import load_dataset  # only needed for Hub downloads
        try:
            ds = load_dataset(source, split="train", columns=NEEDED_COLUMNS)
        except Exception:
            ds = load_dataset(source, split="train")
            ds = ds.select_columns([c for c in NEEDED_COLUMNS if c in ds.column_names])
        df = ds.to_pandas()

    if limit:
        df = df.head(limit)
    if "text" not in df.columns:
        sys.exit("The input has no 'text' column.")
    if "title" not in df.columns:
        df["title"] = ""
    if "article_url" not in df.columns:
        df["article_url"] = df["title"]
    df["text"] = df["text"].fillna("")
    df["title"] = df["title"].fillna("")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Word list for Morfessor
# --------------------------------------------------------------------------
# Very common English function words. Not exhaustive: pass --english-words FILE for a bigger list.
ENGLISH_STOP = set("""
the of and to in is was for with from by on as at are that this it or be which his her their has have
had not were also an who but they been its after first new one two into than other more most some such
only during between about over these those when where while will would could should can may might
""".split())

TOKEN_STRIP = ".,;:!?()[]{}\"'«»*/-_"
WORD_FULL_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
REPEAT_RE = re.compile(r"(.)\1{3,}")


def build_word_counts(lines, min_count, english):
    counts = Counter()
    for line in lines:
        for tok in line.lower().split():
            tok = tok.strip(TOKEN_STRIP)
            if WORD_FULL_RE.fullmatch(tok):
                counts[tok] += 1
    kept = {}
    for w, c in counts.items():
        if c < min_count or not (2 <= len(w) <= 30):
            continue
        if not re.search(r"[aeiou]", w) or re.search(r"[qx]", w) or REPEAT_RE.search(w):
            continue
        if w in english:
            continue
        kept[w] = c
    return kept


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="ffatty/plaintext-wikipedia-full-swahili",
                    help="Parquet file, folder of parquet files, or a Hugging Face dataset id")
    ap.add_argument("--out", default="out", help="Output folder")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N articles (for testing)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.05, help="Validation share, by word count")
    ap.add_argument("--test-frac", type=float, default=0.05, help="Test share, by word count")
    ap.add_argument("--min-article-words", type=int, default=20)
    ap.add_argument("--min-line-words", type=int, default=3)
    ap.add_argument("--max-digit-ratio", type=float, default=0.15, help="Drop articles above this digit share")
    ap.add_argument("--min-alpha-ratio", type=float, default=0.6, help="Drop articles below this letter share")
    ap.add_argument("--max-line-repeat", type=int, default=10,
                    help="Drop a line if it appears in this many articles or more (boilerplate)")
    ap.add_argument("--keep-non-latin", action="store_true", help="Do not strip Arabic/Greek/etc. characters")
    ap.add_argument("--keep-overlap", action="store_true",
                    help="Do not remove validation/test lines that also appear in train")
    ap.add_argument("--min-word-count", type=int, default=2, help="Min frequency for the Morfessor word list")
    ap.add_argument("--english-words", default=None, help="Optional file with one English word per line to exclude")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args), "dropped": Counter()}

    # 1. Load -------------------------------------------------------------
    print("Loading...")
    df = load_articles(args.input, args.limit)
    report["articles_loaded"] = len(df)
    print(f"  {len(df):,} articles")

    # 2. Clean ------------------------------------------------------------
    print("Cleaning articles...")
    cleaned = []
    for i, text in enumerate(df["text"]):
        cleaned.append(clean_text(
            text, strip_non_latin=not args.keep_non_latin,
            min_line_words=args.min_line_words,
        ))
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1:,} / {len(df):,}")
    df["lines"] = cleaned
    df = df.drop(columns=["text"])

    # 3. Drop boilerplate lines (same line in many different articles) -----
    print("Removing boilerplate lines...")
    article_counts = Counter()
    for lines in df["lines"]:
        article_counts.update({line_hash(l) for l in lines})
    boiler = {h for h, c in article_counts.items() if c >= args.max_line_repeat}
    before = sum(len(l) for l in df["lines"])
    df["lines"] = [[l for l in lines if line_hash(l) not in boiler] for lines in df["lines"]]
    report["boilerplate_lines_removed"] = before - sum(len(l) for l in df["lines"])
    print(f"  removed {report['boilerplate_lines_removed']:,} lines")

    # 4. Article-level filters --------------------------------------------
    print("Filtering articles...")
    stats = pd.DataFrame([article_ratios(l) for l in df["lines"]])
    df = pd.concat([df, stats], axis=1)
    reasons = pd.Series("", index=df.index)
    reasons[df["n_words"] == 0] = "empty"
    reasons[(reasons == "") & (df["n_words"] < args.min_article_words)] = "too_short"
    reasons[(reasons == "") & (df["digit_ratio"] > args.max_digit_ratio)] = "digit_heavy"
    reasons[(reasons == "") & (df["alpha_ratio"] < args.min_alpha_ratio)] = "low_letter_share"
    for reason, n in reasons[reasons != ""].value_counts().items():
        report["dropped"][reason] += int(n)
    df = df[reasons == ""].reset_index(drop=True)

    # 5. Remove duplicates ------------------------------------------------
    print("Removing duplicates...")
    df["text_hash"] = [line_hash("\n".join(l)) for l in df["lines"]]
    dup = df.duplicated("text_hash", keep="first")
    report["dropped"]["duplicate_text"] = int(dup.sum())
    df = df[~dup]
    # Same title twice: keep the longer article
    df = df.sort_values("n_words", ascending=False)
    has_title = df["title"].str.len() > 0
    dup_title = has_title & df.duplicated("title", keep="first")
    report["dropped"]["duplicate_title"] = int(dup_title.sum())
    df = df[~dup_title]
    # Same URL twice (even with different text or title): keep the longer article
    has_url = df["article_url"].str.len() > 0
    dup_url = has_url & df.duplicated("article_url", keep="first")
    report["dropped"]["duplicate_url"] = int(dup_url.sum())
    df = df[~dup_url].sort_index().reset_index(drop=True)
    report["articles_after_cleaning"] = len(df)
    print(f"  {len(df):,} articles remain")

    if len(df) < 10:
        sys.exit("Too few articles left to split. Loosen the filters or use more data.")

    # 6. Split by whole article, balanced by word count -------------------
    print("Splitting by article...")
    order = list(df.index)
    random.Random(args.seed).shuffle(order)
    total_words = int(df["n_words"].sum())
    test_target, val_target = args.test_frac * total_words, args.val_frac * total_words
    assign, test_w, val_w = {}, 0, 0
    for idx in order:
        w = int(df.at[idx, "n_words"])
        if test_w < test_target:
            assign[idx] = "test"; test_w += w
        elif val_w < val_target:
            assign[idx] = "validation"; val_w += w
        else:
            assign[idx] = "train"
    df["split"] = pd.Series(assign)

    # 7. Leakage guard: drop val/test lines that also appear in train -----
    train_hashes = {line_hash(l) for lines in df.loc[df["split"] == "train", "lines"] for l in lines}
    overlap = {}
    for split in ("validation", "test"):
        total_lines = overlap_lines = 0
        new_lines = []
        for lines in df.loc[df["split"] == split, "lines"]:
            total_lines += len(lines)
            kept = [l for l in lines if line_hash(l) not in train_hashes]
            overlap_lines += len(lines) - len(kept)
            new_lines.append(kept if not args.keep_overlap else lines)
        overlap[split] = {"lines": total_lines, "also_in_train": overlap_lines}
        if not args.keep_overlap:
            df.loc[df["split"] == split, "lines"] = pd.Series(new_lines, index=df.index[df["split"] == split])
    report["line_overlap_with_train"] = overlap
    if not args.keep_overlap:
        stats = pd.DataFrame([article_ratios(l) for l in df["lines"]], index=df.index)
        df[["n_words", "digit_ratio", "alpha_ratio"]] = stats
        small = df["n_words"] < args.min_article_words
        report["dropped"]["too_short_after_overlap_removal"] = int(small.sum())
        df = df[~small].reset_index(drop=True)

    # Sanity checks
    # Final safety net: no article may appear twice. Rows with no URL or title get a row-based id.
    ids = df["article_url"].where(df["article_url"].str.len() > 0, df["title"])
    ids = ids.where(ids.str.len() > 0, pd.Series([f"row{i}" for i in df.index], index=df.index))
    df["article_url"] = ids
    still_dup = df.duplicated("article_url", keep="first")
    if still_dup.any():
        report["dropped"]["duplicate_id_final"] = int(still_dup.sum())
        df = df[~still_dup].reset_index(drop=True)

    # 8. Write outputs ----------------------------------------------------
    print("Writing outputs...")
    report["splits"] = {}
    for split in ("train", "validation", "test"):
        part = df[df["split"] == split]
        texts = ["\n".join(l) for l in part["lines"]]
        pd.DataFrame({"id": part["article_url"].values, "title": part["title"].values, "text": texts}) \
            .to_parquet(out / f"{split}.parquet", index=False)
        with open(out / f"{split}.txt", "w", encoding="utf-8") as f:
            for lines in part["lines"]:
                for l in lines:
                    f.write(l + "\n")
        report["splits"][split] = {
            "articles": int(len(part)),
            "words": int(part["n_words"].sum()),
            "avg_words_per_article": round(float(part["n_words"].mean()), 1) if len(part) else 0,
            "avg_digit_ratio": round(float(part["digit_ratio"].mean()), 4) if len(part) else 0,
        }

    english = set(ENGLISH_STOP)
    if args.english_words:
        english |= {w.strip().lower() for w in open(args.english_words, encoding="utf-8") if w.strip()}
    train_lines = (l for lines in df.loc[df["split"] == "train", "lines"] for l in lines)
    wc = build_word_counts(train_lines, args.min_word_count, english)
    with open(out / "train_wordcounts.txt", "w", encoding="utf-8") as f:
        for w, c in sorted(wc.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{c} {w}\n")
    report["morfessor_word_types"] = len(wc)

    report["dropped"] = dict(report["dropped"])
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("\nDone. Summary:")
    print(f"  loaded:            {report['articles_loaded']:,}")
    print(f"  dropped:           {report['dropped']}")
    print(f"  boilerplate lines: {report['boilerplate_lines_removed']:,}")
    for split, s in report["splits"].items():
        print(f"  {split:<11} {s['articles']:>7,} articles  {s['words']:>11,} words")
    print(f"  line overlap with train (removed unless --keep-overlap): {overlap}")
    print(f"  Morfessor word types: {len(wc):,}")
    print(f"Files written to {out.resolve()}")


if __name__ == "__main__":
    main()