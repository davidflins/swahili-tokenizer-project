
import re
import random
import argparse
import unicodedata
from pathlib import Path


# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize_unicode(text):
    """Apply NFC normalization and unify quotes and dashes."""

    text = unicodedata.normalize("NFC", text)

    # Normalize quotation marks and apostrophes
    quote_map = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u02bc": "'",
        "\u02bb": "'",
    }

    for old, new in quote_map.items():
        text = text.replace(old, new)

    # Normalize dashes
    dashes = [
        "\u2010", "\u2011", "\u2012",
        "\u2013", "\u2014", "\u2015",
        "\u2212"
    ]

    for dash in dashes:
        text = text.replace(dash, "-")

    return text


# ==========================================
# TEXT CLEANING AND TOKENIZATION
# ==========================================

def clean_text(text, max_word_length=30):

    # 1. NFC normalization and quote/dash unification
    text = normalize_unicode(text)

    # 2. Remove URLs
    text = re.sub(
        r"(?:https?://|www\.)\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # 3. Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # 4. Lowercase everything
    text = text.lower()

    # 5. Tokenize into Unicode alphabetic words.
    # Preserve apostrophes only when they occur internally.
    words = re.findall(
        r"[^\W\d_]+(?:'[^\W\d_]+)*",
        text,
        flags=re.UNICODE
    )

    # 6. Remove excessively long words
    words = [
        word for word in words
        if len(word) <= max_word_length
    ]

    # 7. Remove numeric-only and symbol-only tokens
    # (Already excluded by the alphabetic tokenizer.)

    return " ".join(words)


# ==========================================
# DOCUMENT LOADING
# ==========================================

def load_documents(input_file):
    """Load documents separated by blank lines."""

    text = Path(input_file).read_text(encoding="utf-8")

    documents = re.split(r"\n\s*\n+", text)

    return [
        doc.strip()
        for doc in documents
        if doc.strip()
    ]


# ==========================================
# DATASET SPLITTING
# ==========================================

def split_documents(
    documents,
    train_ratio,
    val_ratio,
    test_ratio,
    seed
):

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0")

    random.seed(seed)
    random.shuffle(documents)

    total = len(documents)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = documents[:train_end]
    val = documents[train_end:val_end]
    test = documents[val_end:]

    return train, val, test


# ==========================================
# SAVE DATASETS
# ==========================================

def save_documents(documents, output_file):

    Path(output_file).write_text(
        "\n\n".join(documents),
        encoding="utf-8"
    )


# ==========================================
# MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Clean and split a Swahili text dataset."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input TXT dataset."
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory."
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.90,
        help="Training split ratio (default: 0.90)."
    )

    parser.add_argument(
        "--val",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05)."
    )

    parser.add_argument(
        "--test",
        type=float,
        default=0.05,
        help="Test split ratio (default: 0.05)."
    )

    parser.add_argument(
        "--max-word-length",
        type=int,
        default=30,
        help="Maximum word length (default: 30)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)."
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    if args.max_word_length < 1:
        parser.error("--max-word-length must be positive")

    if min(args.train, args.val, args.test) < 0:
        parser.error("Split ratios cannot be negative")

    if abs(args.train + args.val + args.test - 1.0) > 1e-8:
        parser.error("Split ratios must sum to 1.0")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load documents
    print("\nLoading dataset...")

    documents = load_documents(input_path)

    print(f"Original documents: {len(documents):,}")

    # Clean documents
    print("\nCleaning documents...")

    cleaned_documents = []

    for doc in documents:

        cleaned = clean_text(
            doc,
            max_word_length=args.max_word_length
        )

        if cleaned:
            cleaned_documents.append(cleaned)

    removed = len(documents) - len(cleaned_documents)

    print(f"Documents after cleaning: {len(cleaned_documents):,}")
    print(f"Empty documents removed: {removed:,}")

    if not cleaned_documents:
        parser.error("No documents remain after cleaning.")

    # Split dataset
    print("\nSplitting dataset...")

    train, val, test = split_documents(
        cleaned_documents,
        args.train,
        args.val,
        args.test,
        args.seed
    )

    # Save datasets
    print("\nSaving datasets...")

    save_documents(train, output_dir / "train.txt")
    save_documents(val, output_dir / "val.txt")
    save_documents(test, output_dir / "test.txt")

    # Print statistics
    print("\n" + "=" * 45)
    print("PREPROCESSING COMPLETE")
    print("=" * 45)

    print(f"Total documents:      {len(cleaned_documents):,}")
    print(f"Training documents:   {len(train):,}")
    print(f"Validation documents: {len(val):,}")
    print(f"Test documents:       {len(test):,}")

    print(f"\nOutput directory: {output_dir.resolve()}")
    print("=" * 45)


if __name__ == "__main__":
    main()
