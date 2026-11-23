
from Tokenizer import ByteLevelBPETokenizer


def run_tests():

    corpus = [
        "machine learning",
        "deep learning",
        "transformers are powerful",
        "machine learning is interesting",
        "hello world",
        "hello machine",
        "Python engineering",
        "learning AI",
        "مرحبا بالعالم",
        "你好世界",
        "🚀 AI engineering",
    ] * 10

    tokenizer = ByteLevelBPETokenizer(
        vocab_size=500,
        min_frequency=2,
    )

    tokenizer.train(
        corpus,
        verbose=False,
    )

    test_cases = [
        "Hello world!",
        "Machine learning is powerful.",
        "مرحبا بالعالم",
        "你好世界",
        "🚀🔥",
        "Multiple     spaces",
        "New\nlines\nare important",
        "",
        "Unseen vocabulary: supercalifragilisticexpialidocious",
    ]

    for text in test_cases:

        ids = tokenizer.encode(text)

        decoded = tokenizer.decode(ids)

        assert decoded == text, (
            f"Round-trip failed: {text!r} != {decoded!r}"
        )

        print("PASS:", ascii(text))

    # Special token test
    ids = tokenizer.encode(
        "Hello",
        add_special_tokens=True,
    )

    assert ids[0] == tokenizer.special_to_id["[BOS]"]
    assert ids[-1] == tokenizer.special_to_id["[EOS]"]
    assert len(set(tokenizer.special_to_id.values())) == len(
        tokenizer.special_tokens
    )

    assert tokenizer.decode(ids) == "Hello"

    assert (
        tokenizer.decode(
            ids,
            skip_special_tokens=False
        )
        == "[BOS]Hello[EOS]"
    )

    # Persistence test
    tokenizer.save("test_tokenizer.json")

    loaded = ByteLevelBPETokenizer.load(
        "test_tokenizer.json"
    )

    for text in test_cases:

        assert tokenizer.encode(text) == loaded.encode(text)

        assert tokenizer.decode(
            loaded.encode(text)
        ) == text

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
