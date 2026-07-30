import torch
import torch.nn as nn

from model import GPT
from data import load_data

# ---------------- hyperparameters ----------------

# Model
context_length = 8
embed_dim = 32
n_layers = 4
num_heads = 4

# Training:
batch_size = 4
learning_rate = 1e-3
epochs = 1000

# Dataset
stride = context_length
shuffle = True
drop_last=True
num_workers=0

# -------------------------------------------------


vocab, tokenizer, dataloader = load_data(batch_size, context_length, stride, shuffle, drop_last, num_workers)
vocab_size = len(vocab)

model = GPT(vocab_size=vocab_size, T=context_length, C=embed_dim ,n_layers=n_layers, num_head=num_heads)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for xb, yb in dataloader:
        optimizer.zero_grad()
        logits = model(xb)     # (B, T, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}  Loss: {loss.item():.4f}")

torch.save({
    'model_args': {
        'vocab_size': vocab_size,
        'T': context_length,
        'C': embed_dim,
        'n_layers': n_layers,
        'num_head': num_heads,
    },
    'vocab': vocab,
    'state_dict': model.state_dict(),
}, 'model.pt')