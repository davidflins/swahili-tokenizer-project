#!/usr/bin/env python3
"""
Morfessor-first tokenizer with BPE *inside* morphemes (merges never cross a morph boundary),
plus rule-based fallback layers for English, numbers, URLs/emails and acronyms.

How a piece of text is tokenized
--------------------------------
Text is normalized (NFC, unified quotes/dashes/apostrophes) and split into pre-tokens.
Each pre-token is routed to exactly one layer:

  Swahili word  -> Morfessor segments it into morphs; BPE is run on each morph separately,
                   so no token can span two morphs.
  English/other -> English-looking words (word list + spelling heuristics) and words with
                   non-a-z letters skip Morfessor and go straight to BPE.
  Number        -> one token per digit (separators like "," and "." are their own tokens).
  URL / email   -> split at punctuation; letter chunks use BPE, digits and symbols are single
                   characters, case is preserved.
  Acronym       -> ALL-CAPS words of 2..N letters (NATO, CCM, U.S.A.) become one token per letter.
  Other         -> punctuation, symbols and emoji are single characters.

Anything outside the character vocabulary falls back to UTF-8 byte tokens, so nothing is
ever unknown. A word-start marker "▁" is attached to the first piece of each word (and
emitted as its own token before numbers, URLs, acronyms and symbols).

Everything except acronyms and URLs/emails is lowercased, so decode(encode(x)) returns the
*canonical* form of x (see `canonical`), not necessarily the exact original.

Commands
--------
  train     learn the BPE merges from a text corpus, using your Morfessor .bin model
  encode    tokenize text and print the tokens (or ids)
  selftest  round-trip check + statistics (tokens per word, byte-fallback rate, per layer)

Examples
--------
  python morph_bpe_tokenizer.py train --corpus train.txt --morfessor sw_morfessor.bin \
         --vocab-size 16000 --out tokenizer.json
  python morph_bpe_tokenizer.py encode --tokenizer tokenizer.json --text "Wanafunzi 25 wa UDSM wanasoma."
  python morph_bpe_tokenizer.py selftest --tokenizer tokenizer.json --file validation.txt

Requires: morfessor (pip install morfessor)
"""

import argparse
import heapq
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SP = "\u2581"                                   # word-start marker
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]
BYTE_RE = re.compile(r"<0x([0-9A-F]{2})>")
ASCII_PRINTABLE = [chr(c) for c in range(33, 127)]

CHAR_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u02bc": "'", "\u02bb": "'", "`": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u00a0": " ",
})

PRETOKEN_RE = re.compile(
    r"(?P<url>(?:https?://|ftp://|www\.)\S*[^\s.,;:!?)\"'\]]"
    r"|(?<![\w.+\-])[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+)"
    r"|(?P<num>\d+(?:[.,]\d+)*)"
    r"|(?P<acr>(?:[A-Z]\.){2,}|\b[A-Z]{2,}\b)"
    r"|(?P<word>[^\W\d_]+(?:'[^\W\d_]+)*)"
    r"|(?P<other>\S)"
)
URL_CHUNK_RE = re.compile(r"[A-Za-z]+|[^A-Za-z]")
SW_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
# Spellings that are essentially never native Swahili
ENGLISH_HEURISTIC_RE = re.compile(r"[qx]|c(?!h)|tion|sion|ough|ight|ck|ph")

# Very common English function words (extend with --english-words FILE)
BUILTIN_ENGLISH = set("""
the of and to in is was for with from by on as at are that this it or be which his her their has have
had not were also an who but they been its after first new one two into than other more most some such
only during between about over these those when where while will would could should can may might
you your we our he she them there here what how why all any each if then so do does did just very
""".split())


def normalize_text(text):
    text = unicodedata.normalize("NFC", text).translate(CHAR_MAP)
    return text.replace(SP, " ")


# --------------------------------------------------------------------------
# BPE training (merges only ever happen inside a single piece)
# --------------------------------------------------------------------------

def train_bpe(piece_counts, num_merges, min_count=2):
    words = [list(p) for p in piece_counts]
    freqs = list(piece_counts.values())
    pair_counts = defaultdict(int)
    where = defaultdict(set)
    for i, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_counts[p] += freqs[i]
            where[p].add(i)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges = []

    while heap and len(merges) < num_merges:
        negc, pair = heapq.heappop(heap)
        c = pair_counts.get(pair, 0)
        if c < min_count:
            continue
        if -negc != c:                       # stale entry: refresh and retry
            heapq.heappush(heap, (-c, pair))
            continue
        a, b = pair
        new = a + b
        merges.append(pair)
        for i in list(where.pop(pair, ())):
            w, f = words[i], freqs[i]
            nw, j = [], 0
            while j < len(w):
                if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                    nw.append(new); j += 2
                else:
                    nw.append(w[j]); j += 1
            if len(nw) == len(w):
                continue
            old, newc = Counter(zip(w, w[1:])), Counter(zip(nw, nw[1:]))
            for p in old.keys() | newc.keys():
                delta = (newc.get(p, 0) - old.get(p, 0)) * f
                if delta:
                    pair_counts[p] += delta
                    if delta > 0:
                        heapq.heappush(heap, (-pair_counts[p], p))
                if p not in newc:
                    s = where.get(p)
                    if s:
                        s.discard(i)
                elif p not in old:
                    where[p].add(i)
            words[i] = nw
        pair_counts.pop(pair, None)
    return merges


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

class MorphBPETokenizer:
    def __init__(self, config, english_words, morfessor_path=None, vocab=None, merges=None):
        self.cfg = config
        self.english = set(english_words)
        self.morfessor_path = morfessor_path
        self.model = None
        self.tokens = list(vocab or [])
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self._seg_cache, self._bpe_cache = {}, {}

    # ---- persistence ----
    def save(self, path):
        data = {
            "format": "morph-bpe-v1", "config": self.cfg,
            "morfessor_model": str(self.morfessor_path) if self.morfessor_path else None,
            "english_words": sorted(self.english),
            "vocab": self.tokens, "merges": [list(m) for m in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path, morfessor_path=None):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["config"], d["english_words"], morfessor_path or d.get("morfessor_model"),
                   d["vocab"], d["merges"])

    # ---- Morfessor ----
    def _get_model(self):
        if self.model is None:
            if not self.morfessor_path or not Path(self.morfessor_path).exists():
                raise RuntimeError(f"Morfessor model not found: {self.morfessor_path!r}. "
                                   "Pass --morfessor path/to/model.bin")
            import morfessor
            self.model = morfessor.MorfessorIO().read_binary_model_file(str(self.morfessor_path))
        return self.model

    def segment(self, word):
        seg = self._seg_cache.get(word)
        if seg is None:
            seg = list(self._get_model().viterbi_segment(word)[0])
            if "".join(seg) != word:         # safety: never lose characters
                seg = [word]
            self._seg_cache[word] = seg
        return seg

    # ---- routing ----
    def is_english(self, word):
        if word in self.english:
            return True
        if self.cfg.get("english_heuristics", True):
            if ENGLISH_HEURISTIC_RE.search(word):
                return True
            if len(word) >= 4 and word[-1] not in "aeiou'":     # native Swahili words end in vowels
                return True
        return False

    def is_swahili(self, word):
        return bool(SW_WORD_RE.fullmatch(word)) and not self.is_english(word)

    def iter_pretokens(self, text, normalized=False):
        if not normalized:
            text = normalize_text(text)
        prev_end, first = 0, True
        for m in PRETOKEN_RE.finditer(text):
            kind, s = m.lastgroup, m.group()
            space = first or m.start() > prev_end
            first, prev_end = False, m.end()
            if kind == "acr":
                if sum(c.isalpha() for c in s) > self.cfg["max_acronym_len"]:
                    kind, s = "word", s.lower()      # shouting, not an acronym
            elif kind == "word":
                s = s.lower()
            yield kind, s, space

    # ---- encoding ----
    def _chars(self, s):
        out = []
        for ch in s:
            if ch in self.vocab:
                out.append(ch)
            else:
                out.extend(f"<0x{b:02X}>" for b in ch.encode("utf-8"))
        return out

    def _bpe(self, piece):
        cached = self._bpe_cache.get(piece)
        if cached is not None:
            return cached
        sym = self._chars(piece)
        while len(sym) > 1:
            best, best_rank = None, None
            for i in range(len(sym) - 1):
                r = self.ranks.get((sym[i], sym[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = (sym[i], sym[i + 1]), r
            if best is None:
                break
            merged, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    merged.append(best[0] + best[1]); i += 2
                else:
                    merged.append(sym[i]); i += 1
            sym = merged
        self._bpe_cache[piece] = sym
        return sym

    def encode_pretoken(self, kind, s, space):
        if kind == "word":
            prefix = SP if space else ""
            if self.is_swahili(s):
                out = []
                for k, morph in enumerate(self.segment(s)):
                    out += self._bpe((prefix if k == 0 else "") + morph)   # BPE inside each morph
                return out
            return self._bpe(prefix + s)                                   # English / other scripts
        out = [SP] if space else []
        if kind == "url":
            for chunk in URL_CHUNK_RE.findall(s):
                out += self._bpe(chunk) if chunk.islower() else self._chars(chunk)
        else:                                                              # num, acr, other
            out += self._chars(s)
        return out

    def label(self, kind, s):
        if kind == "word":
            return "swahili" if self.is_swahili(s) else "english/other"
        return {"num": "number", "url": "url/email", "acr": "acronym", "other": "symbol"}[kind]

    def encode(self, text):
        out = []
        for kind, s, space in self.iter_pretokens(text):
            out += self.encode_pretoken(kind, s, space)
        return out

    def encode_ids(self, text):
        unk = self.vocab.get("<unk>")
        return [self.vocab.get(t, unk) for t in self.encode(text)]

    def canonical(self, text):
        parts = []
        for kind, s, space in self.iter_pretokens(text):
            parts.append((" " if space and parts else "") + s)
        return "".join(parts)

    def decode(self, tokens):
        out, buf = [], bytearray()

        def flush():
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf.clear()

        for t in tokens:
            if t in SPECIALS:
                continue
            m = BYTE_RE.fullmatch(t)
            if m:
                buf.append(int(m.group(1), 16))
            else:
                flush()
                out.append(t)
        flush()
        s = "".join(out).replace(SP, " ")
        return s[1:] if s.startswith(" ") else s

    def decode_ids(self, ids):
        return self.decode([self.tokens[i] for i in ids])


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def read_lines(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def cmd_train(a):
    english = set(BUILTIN_ENGLISH)
    if a.english_words:
        english |= {w.strip().lower() for w in open(a.english_words, encoding="utf-8") if w.strip()}
    cfg = {"max_acronym_len": a.max_acronym_len, "english_heuristics": not a.no_english_heuristics}
    tok = MorphBPETokenizer(cfg, english, a.morfessor)
    tok._get_model()                                         # fail early if the .bin is unreadable

    print("Counting words...")
    word_counts, char_counts = Counter(), Counter()
    for n, line in enumerate(read_lines(a.corpus), 1):
        line = normalize_text(line)
        char_counts.update(line)
        for kind, s, _ in tok.iter_pretokens(line, normalized=True):
            if kind == "word":
                word_counts[s] += 1
        if n % 200000 == 0:
            print(f"  {n:,} lines")
    print(f"  {len(word_counts):,} distinct words")

    # Character vocabulary: specials, bytes, marker, ASCII, plus frequent non-ASCII characters
    extras = sorted(c for c, n in char_counts.items()
                    if n >= a.min_char_count and ord(c) > 126 and c.isprintable() and not c.isspace())
    base = SPECIALS + BYTE_TOKENS + [SP] + ASCII_PRINTABLE + extras
    base_chars = set([SP] + ASCII_PRINTABLE + extras)
    if a.vocab_size <= len(base):
        sys.exit(f"--vocab-size must be larger than the base vocabulary ({len(base)}).")

    print("Segmenting with Morfessor and building pieces...")
    piece_counts, skipped, n_sw, n_en = Counter(), 0, 0, 0
    for w, c in word_counts.items():
        if tok.is_swahili(w):
            n_sw += 1
            pieces = [(SP if k == 0 else "") + m for k, m in enumerate(tok.segment(w))]
        else:
            n_en += 1
            pieces = [SP + w]
        for p in pieces:
            if all(ch in base_chars for ch in p):
                piece_counts[p] += c
            else:
                skipped += 1
    print(f"  {n_sw:,} Swahili-routed words, {n_en:,} English/other words, "
          f"{len(piece_counts):,} distinct pieces ({skipped} skipped for rare characters)")

    num_merges = a.vocab_size - len(base)
    print(f"Training BPE inside pieces ({num_merges:,} merges requested)...")
    merges = train_bpe(piece_counts, num_merges, a.min_merge_count)

    vocab = list(base)
    seen = set(vocab)
    for x, y in merges:
        if x + y not in seen:
            seen.add(x + y)
            vocab.append(x + y)
    tok = MorphBPETokenizer(cfg, english, a.morfessor, vocab, merges)
    tok.save(a.out)
    print(f"Learned {len(merges):,} merges; vocabulary size {len(vocab):,}. Saved to {a.out}")
    if len(merges) < num_merges:
        print("  (fewer merges than requested: remaining pairs were too rare. Lower --min-merge-count "
              "or --vocab-size.)")


def cmd_encode(a):
    tok = MorphBPETokenizer.load(a.tokenizer, a.morfessor)
    texts = [a.text] if a.text else list(read_lines([a.file]))
    for t in texts:
        if a.ids:
            print(tok.encode_ids(t))
        else:
            print(" | ".join(tok.encode(t)))


def cmd_selftest(a):
    tok = MorphBPETokenizer.load(a.tokenizer, a.morfessor)
    lines = []
    for line in read_lines([a.file]):
        lines.append(line)
        if len(lines) >= a.max_lines:
            break
    fails, total_tokens, total_words, byte_tokens = [], 0, 0, 0
    per_layer = defaultdict(lambda: [0, 0])                 # label -> [pretokens, tokens]
    for line in lines:
        toks = tok.encode(line)
        if tok.decode(toks) != tok.canonical(line):
            fails.append(line)
        total_tokens += len(toks)
        byte_tokens += sum(1 for t in toks if BYTE_RE.fullmatch(t))
        total_words += len(tok.canonical(line).split())
        for kind, s, space in tok.iter_pretokens(line):
            lab = tok.label(kind, s)
            per_layer[lab][0] += 1
            per_layer[lab][1] += len(tok.encode_pretoken(kind, s, space))
    print(f"Lines checked:            {len(lines):,}")
    print(f"Round-trip failures:      {len(fails)}")
    for f in fails[:3]:
        print(f"   e.g. {f[:100]!r}")
    print(f"Tokens per word:          {total_tokens / max(total_words, 1):.2f}")
    print(f"Byte-fallback token rate: {100 * byte_tokens / max(total_tokens, 1):.3f}%")
    print("Per layer (pre-tokens, tokens, tokens each):")
    for lab, (n, t) in sorted(per_layer.items()):
        print(f"   {lab:<14} {n:>9,} {t:>10,}   {t / max(n, 1):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Learn BPE merges using a Morfessor model")
    t.add_argument("--corpus", nargs="+", required=True, help="Text file(s), one paragraph per line")
    t.add_argument("--morfessor", required=True, help="Path to the Morfessor .bin model")
    t.add_argument("--out", default="tokenizer.json")
    t.add_argument("--vocab-size", type=int, default=16000)
    t.add_argument("--english-words", help="Optional file with one English word per line")
    t.add_argument("--no-english-heuristics", action="store_true",
                   help="Only use the word lists to detect English (no spelling rules)")
    t.add_argument("--max-acronym-len", type=int, default=6)
    t.add_argument("--min-char-count", type=int, default=20,
                   help="Non-ASCII characters rarer than this fall back to bytes")
    t.add_argument("--min-merge-count", type=int, default=2)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("encode", help="Tokenize text")
    e.add_argument("--tokenizer", required=True)
    e.add_argument("--morfessor", help="Override the Morfessor path stored in the tokenizer file")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    e.add_argument("--ids", action="store_true")
    e.set_defaults(func=cmd_encode)

    s = sub.add_parser("selftest", help="Round-trip check and statistics on a text file")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--morfessor")
    s.add_argument("--file", required=True)
    s.add_argument("--max-lines", type=int, default=5000)
    s.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
