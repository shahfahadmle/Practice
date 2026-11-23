
from pathlib import Path

from Tokenizer import ByteLevelBPETokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "input.txt"
OUTPUT_PATH = BASE_DIR / "tokenizer.json"


def load_training_data(path: Path):

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:

        for line in file:
            line = line.rstrip("\r\n")

            if line:
                yield line


def main():

    tokenizer = ByteLevelBPETokenizer(
        vocab_size=16000,
        min_frequency=2,
    )

    tokenizer.train(
        texts=load_training_data(DATA_PATH),
        verbose=True,
    )

    tokenizer.save(OUTPUT_PATH)

    print("\nTokenizer saved.")

    sample = "I am learning Byte Pair Encoding."

    ids = tokenizer.encode(
        sample,
        add_special_tokens=True,
    )

    print("Original:", sample)
    print("Token IDs:", ids)
    print("Token count:", len(ids))

    print(
        "Decoded:",
        tokenizer.decode(ids)
    )


if __name__ == "__main__":
    main()
