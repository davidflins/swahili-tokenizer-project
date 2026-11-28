import csv
input_file = "C:/Users/David Lins/freshmansem/swahili-tokenizer-project/data/raw/semeval/val.tsv"
output_file = "C:/Users/David Lins/freshmansem/swahili-tokenizer-project/data/raw/semeval/val.txt"

with open(input_file, "r", newline="", encoding="utf-8") as infile:
    with open(output_file, "w", encoding="utf-8") as outfile:
        reader = csv.reader(infile, delimiter="\t")

        for row in reader:
            if row:  # Skip empty rows
                outfile.write(row[0] + "\n")

print(f"First column extracted to {output_file}")
