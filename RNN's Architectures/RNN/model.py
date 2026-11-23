import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(nn.Module):
    def __init__(self, vocab_size, embeding_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embeding_dim,
        )
        self.rnn = nn.RNN(
            input_size=embeding_dim,
            hidden_size=hid_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=hid_dim, out_features=1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, length):
        if x.ndim != 2:
            raise ValueError("Input reviews must have shape (batch, sequence).")
        if length.ndim != 1 or length.size(0) != x.size(0):
            raise ValueError("Review lengths must contain one value per review.")
        if torch.any(length <= 0):
            raise ValueError("Reviews must contain at least one token.")

        embedded = self.embedding(x)
        packed = pack_padded_sequence(
            embedded,
            length.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden).squeeze(1)
