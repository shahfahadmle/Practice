
from Tokenizer import ByteLevelBPETokenizer


def main():

    tokenizer = ByteLevelBPETokenizer.load(
        "tokenizer.json"
    )

    text = """
    Transformers are powerful.
    I am learning modern LLM engineering.
    """

    # Encoding
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
    )

    # Decoding
    reconstructed = tokenizer.decode(
        token_ids
    )

    print("Original text:")
    print(text)

    print("\nToken IDs:")
    print(token_ids)

    print("\nToken count:")
    print(len(token_ids))

    print("\nReconstructed:")
    print(reconstructed)


if __name__ == "__main__":
    main()
