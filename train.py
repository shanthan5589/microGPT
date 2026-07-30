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
epochs = 1
learning_rate = 1e-3
batch_size = 4

# Dataset
stride = context_length
shuffle = True
drop_last=True
num_workers=0

# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

vocab, tokenizer, dataloader = load_data(batch_size, context_length, stride, shuffle, drop_last, num_workers)
vocab_size = len(vocab)

model = GPT(vocab_size=vocab_size, T=context_length, C=embed_dim ,n_layers=n_layers, num_head=num_heads).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
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