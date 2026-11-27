
import re
import random
import argparse
import unicodedata
from pathlib import Path


def clean_text(text, max_word_length=30):
    """Clean and normalize text."""

    # 1. NFC Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize quotation marks and apostrophes
    quote_map = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }

    for old, new in quote_map.items():
        text = text.replace(old, new)

    # 3. Normalize dashes
    dashes = [
        "\u2010", "\u2011", "\u2012",
        "\u2013", "\u2014", "\u2015",
        "\u2212"
    ]

    for dash in dashes:
        text = text.replace(dash, "-")

    # 4. Remove URLs
    text = re.sub(
        r"(?:https?://|www\.)\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # 5. Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # 6. Lowercase
    text = text.lower()

    # 7. Remove very long words
    words = text.split()
    words = [
        word for word in words
        if len(word) <= max_word_length
    ]

    text = " ".join(words)

    # 8. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_documents(input_file):
    """Load documents separated by blank lines."""

    text = Path(input_file).read_text(encoding="utf-8")

    documents = re.split(r"\n\s*\n+", text)

    return [
        doc.strip()
        for doc in documents
        if doc.strip()
    ]


def split_documents(documents, train_ratio, val_ratio, test_ratio, seed):
    """Shuffle and split documents."""

    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8:
        raise ValueError("Split ratios must add up to 1.0")

    random.seed(seed)
    random.shuffle(documents)

    total = len(documents)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = documents[:train_end]
    val = documents[train_end:val_end]
    test = documents[val_end:]

    return train, val, test


def save_documents(documents, output_file):
    """Save documents separated by blank lines."""

    Path(output_file).write_text(
        "\n\n".join(documents),
        encoding="utf-8"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Clean and split a large text dataset."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input TXT file."
    )

    parser.add_argument(
        "--output",
        "-o",
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
        help="Maximum allowed word length (default: 30)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    if args.max_word_length < 1:
        parser.error("--max-word-length must be positive")

    if min(args.train, args.val, args.test) < 0:
        parser.error("Split ratios cannot be negative")

    if not abs(args.train + args.val + args.test - 1.0) < 1e-8:
        parser.error("Train, validation, and test ratios must sum to 1")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading documents...")

    documents = load_documents(input_path)

    print(f"Loaded documents: {len(documents):,}")

    print("Cleaning documents...")

    cleaned_documents = [
        clean_text(doc, args.max_word_length)
        for doc in documents
    ]

    cleaned_documents = [
        doc for doc in cleaned_documents
        if doc
    ]

    print(f"Documents after cleaning: {len(cleaned_documents):,}")

    print("Splitting dataset...")

    train, val, test = split_documents(
        cleaned_documents,
        args.train,
        args.val,
        args.test,
        args.seed
    )

    # Save output files
    save_documents(train, output_dir / "train.txt")
    save_documents(val, output_dir / "val.txt")
    save_documents(test, output_dir / "test.txt")

    print("\nDataset successfully processed!")
    print("-" * 40)
    print(f"Train:      {len(train):,} documents")
    print(f"Validation: {len(val):,} documents")
    print(f"Test:       {len(test):,} documents")
    print("-" * 40)
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
