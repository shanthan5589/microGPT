import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, T, C):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, C)    # (vocab_size, C)
        self.position_embedding = nn.Embedding(T, C)     # (T, C)

    def forward(self, x):
        B, T = x.shape         
        token = self.token_embedding(x)    # (B, T, C)
        position_ids = torch.arange(T, device=x.device)   # (T, C)
        position = self.position_embedding(position_ids)   # (T, C)
        return token + position           # (B, T, C) Broadcasting works.


class Head(nn.Module):
    def __init__(self, C, T, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(C, head_size, bias=False)
        self.key = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))

    def forward(self, x):
        
        B, T, C = x.shape
        
        q = self.query(x)    # (B, T, head_size)
        k = self.key(x)      # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        wei = q @ k.transpose(-2, -1)  * (self.head_size ** -0.5)        # (B, T, head_size) x (B, head_size, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)

        out = wei @ v    # (B, T, T) x (B, T, head_size) = (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, C, T, num_heads):
        super().__init__()
        self.head_size = C // num_heads
        self.heads = nn.ModuleList([Head(C, T, self.head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4*C),
            nn.ReLU(),
            nn.Linear(4*C, C)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, C, T, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = MultiHeadAttention(C, T, num_heads)
        self.ln2 = nn.LayerNorm(C)
        self.ff = FeedForward(C)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size, T=8, C=32, n_layers=4, num_head=4):
        super().__init__()
        self.T = T
        self.embedding = Embedding(vocab_size, T, C)
        self.blocks = nn.Sequential(*[Block(C, T, num_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_tokens=100, temperature=1.0):
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(1) <= self.T else idx[:, -self.T:]

            logits = self(idx_cond)   # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            probs = torch.softmax(logits, dim=-1)     # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)    # (B)
    
            idx = torch.cat((idx, idx_next), dim=1)

        return idx