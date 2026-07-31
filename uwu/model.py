import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, T, C):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, C)

    def forward(self, x):
        B, T = x.shape
        token_embeddings = self.token_embedding(x)
        out = token_embeddings
        return out

class MLP(nn.Module):
    def __init__(self, vocab_size, T, C):
        super().__init__()
        self.T = T
        self.embedding = Embedding(vocab_size, T, C)
        self.net = nn.Sequential(
            nn.Linear(T * C, 300),
            nn.ReLU(),
            nn.Linear(300, vocab_size)
        )   

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)   # (B, T * C)
        out = self.net(x)    # (B, vocab_size)
        return out