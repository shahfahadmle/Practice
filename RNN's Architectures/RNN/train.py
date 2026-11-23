from collections import Counter
from pathlib import Path
import random

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model import RNN
from preprocessing import tokenize_review


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Dataset" / "IMDB Dataset.csv"
MODEL_PATH = ROOT / "Model" / "imdb_rnn_model.pth"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class ReviewDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        tokens, label = self.examples[index]
        return tokens, torch.tensor(
            label, dtype=torch.float32
        )


def collate_reviews(batch):
    reviews, labels = zip(*batch)
    lengths = torch.tensor([len(review) for review in reviews], dtype=torch.long)
    reviews = pad_sequence(reviews, batch_first=True, padding_value=0)
    return reviews, lengths, torch.stack(labels)


def make_loader(examples, batch_size, shuffle):
    return DataLoader(
        ReviewDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_reviews,
    )


def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_total = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for reviews, lengths, labels in loader:
            reviews = reviews.to(device)
            labels = labels.to(device)
            logits = model(reviews, lengths)
            loss_total += loss_fn(logits, labels).item()
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            count += labels.numel()

    return loss_total / len(loader), correct / count


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "Dataset not found. Expected the IMDB CSV at "
            f"{DATA_PATH}."
        )

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(DATA_PATH)
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must have 'review' and 'sentiment' columns.")
    if data[list(required_columns)].isnull().any().any():
        raise ValueError("Dataset contains missing reviews or sentiment labels.")

    labels = data["sentiment"].map({"positive": 1, "negative": 0})
    if labels.isnull().any():
        raise ValueError("Sentiment labels must be 'positive' or 'negative'.")

    indices = list(range(len(data)))
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=42,
        stratify=labels,
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=1 / 9,
        random_state=42,
        stratify=labels.iloc[train_val_indices],
    )

    print(f"Building vocabulary from {len(train_indices):,} training reviews...")
    word_counts = Counter()
    max_length = 0
    for index in train_indices:
        tokens = tokenize_review(data.iloc[index]["review"])
        if not tokens:
            raise ValueError(f"Review at row {index} is empty after preprocessing.")
        word_counts.update(tokens)
        max_length = max(max_length, len(tokens))

    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in word_counts.most_common(19_998):
        word2idx[word] = len(word2idx)

    def encode_rows(row_indices):
        examples = []
        for index in row_indices:
            tokens = tokenize_review(data.iloc[index]["review"])
            if not tokens:
                raise ValueError(
                    f"Review at row {index} is empty after preprocessing."
                )
            token_ids = [
                word2idx.get(token, word2idx[UNK_TOKEN])
                for token in tokens[:max_length]
            ]
            examples.append(
                (torch.tensor(token_ids, dtype=torch.long), int(labels.iloc[index]))
            )
        return examples

    batch_size = 64
    train_loader = make_loader(
        encode_rows(train_indices), batch_size, shuffle=True
    )
    val_loader = make_loader(
        encode_rows(val_indices), batch_size, shuffle=False
    )
    test_loader = make_loader(
        encode_rows(test_indices), batch_size, shuffle=False
    )

    embedding_dim = 100
    hidden_dim = 128
    model = RNN(len(word2idx), embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_accuracy = -1.0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for reviews, lengths, batch_labels in train_loader:
            reviews = reviews.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(reviews, lengths)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch + 1:02d}/10 | "
            f"train loss {train_loss / len(train_loader):.4f} | "
            f"validation loss {val_loss:.4f} | "
            f"validation accuracy {val_accuracy:.2%}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(word2idx),
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "max_length": max_length,
                    "word2idx": word2idx,
                    "pad_idx": word2idx[PAD_TOKEN],
                    "unk_idx": word2idx[UNK_TOKEN],
                },
                MODEL_PATH,
            )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Best checkpoint saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
