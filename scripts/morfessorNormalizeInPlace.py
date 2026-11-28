import re
import sys
from pathlib import Path


def format_file_in_place(file_path: str) -> int:
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")

    # Lowercase and strip punctuation while keeping Unicode letters/numbers and whitespace
    cleaned = re.sub(r"[^\w\s]", " ", raw_text.lower(), flags=re.UNICODE)
    words = cleaned.split()

    # Write one word per line with \n line endings and no blank lines
    output_text = "\n".join(words) + ("\n" if words else "")
    path.write_text(output_text, encoding="utf-8", newline="\n")
    return len(words)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python format_test_words.py <path_to_file>")
        sys.exit(1)

    count = format_file_in_place(sys.argv[1])
    print(f"Formatted {count} words in place.")