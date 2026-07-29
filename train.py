import torch
import torch.nn as nn

from model import GPT
from data import load_data

# ---------------- hyperparameters ----------------

# Model
T = 8
C = 32
n_layers = 4
num_heads = 4

# Training:
B = 4
lr = 1e-3
epochs = 1000

# Dataset
stride = 1
shuffle = True
drop_last=True
num_workers=0

# -------------------------------------------------


vocab_size, tokenizer, dataloader = load_data(B, T, stride, shuffle, drop_last, num_workers)
model = GPT(vocab_size=vocab_size, T=T, C=C ,n_layers=n_layers, num_head=num_heads)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
        'T': T,
        'C': C,
        'n_layers': n_layers,
        'num_heads': num_heads,
    },
    'state_dict': model.state_dict(),
}, 'model.pt')