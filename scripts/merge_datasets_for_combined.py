from pathlib import Path

# Configuration
source_dirs = [Path("fineWeb"), Path("semeval"), Path("wikipedia")]
filenames = ["train.txt", "val.txt", "test.txt"]
output_dir = Path("combined")

# Create the output folder if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

for fname in filenames:
    out_path = output_dir / fname
    
    # Stream line-by-line so large datasets (like fineWeb/wikipedia) don't exhaust RAM
    with out_path.open("w", encoding="utf-8") as outfile:
        for folder in source_dirs:
            in_path = folder / fname
            if not in_path.exists():
                print(f"Warning: {in_path} not found, skipping.")
                continue
                
            with in_path.open("r", encoding="utf-8") as infile:
                last_char = ""
                for line in infile:
                    outfile.write(line)
                    last_char = line[-1] if line else last_char
                
                # Ensure a clean newline separates the end of one file from the next
                if last_char and last_char != "\n":
                    outfile.write("\n")

    print(f"Created {out_path}")
