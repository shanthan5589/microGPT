import torch
import torch.nn as nn

from model import MLP
from data import load_data

# ---------------- hyperparameters ----------------

# Model
context_length = 4
embed_dim = 8

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

model = MLP(vocab_size=vocab_size, T=context_length, C=embed_dim).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device) 
        optimizer.zero_grad()
        logits = model(xb)     # (B, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch}  Loss: {total_loss/len(dataloader):.4f}")

torch.save({
    'model_args': {
        'vocab_size': vocab_size,
        'T': context_length,
        'C': embed_dim
    },
    'vocab': vocab,
    'state_dict': model.state_dict(),
}, 'uwu.pt')