import numpy as np
import torch
from torch import nn


class WikiModel(nn.Module):
    def __init__(self, charcount, embedding_dim, hidden_size):
        super(WikiModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=charcount,
                                      embedding_dim=embedding_dim)
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                    batch_first=True),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                    batch_first=True),
            nn.LSTM(input_size=hidden_size, hidden_size=embedding_dim,
                    batch_first=True),
        ])
        self.projection = nn.Linear(in_features=embedding_dim,
                                    out_features=charcount)

    def forward(self, input, forward=0):
        h = input  # (n, t)
        h = self.embedding(h)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.projection(h)
        logits = h
        if forward > 0:
            outputs = []
            h = torch.max(logits, dim=2)[1]
            for i in range(forward):
                h1 = self.embedding(h)
                for j, rnn in enumerate(self.rnns):
                    h1, state = rnn(h1, states[j])
                    states[j] = state
                h = self.projection(h1)
                h = torch.max(h, dim=2)[1]
                outputs.append(h[-1, :].data)
            logits = torch.stack(outputs)
        return logits
