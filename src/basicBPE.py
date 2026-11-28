from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

VOCAB_SIZE = 16000
OUT_DIR = Path("artifacts/tokenizer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build the tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

#train from files
files = [str(p) for p in Path("data/raw").glob("*.txt")]
tokenizer.train(files, trainer)


tokenizer.save(str(OUT_DIR / "tokenizer.json"))
print("Saved. Vocab size:", tokenizer.get_vocab_size())