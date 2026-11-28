import sys

if len(sys.argv) < 2:
    print("Usage: python remove_numbers.py <filename>")
    sys.exit(1)

filepath = sys.argv[1]

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(filepath, "w", encoding="utf-8") as f:
    for line in lines:
        if not line.strip().isdigit():
            f.write(line)
