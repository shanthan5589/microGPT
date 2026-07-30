import torch
from model import GPT
from data import Tokenizer

text = " "
max_tokens = 100
temperature = 1.0

def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0):
    model.eval()
    idx = tokenizer.encode(prompt)
    idx = torch.tensor(idx, dtype=torch.long).unsqueeze(0)  # (1, T) because the model expects (B, T)
    output = model.generate(idx, max_tokens=max_tokens, temperature=temperature)
    return tokenizer.decode(output[0].tolist())   # output is (1, T), so we access the first element and convert it to a list of integers before decoding.

if __name__ == "__main__":
    ckpt = torch.load('model.pt')
    model = GPT(**ckpt['model_args'])
    model.load_state_dict(ckpt['state_dict'])
    tokenizer = Tokenizer(ckpt['vocab'])  # Assuming the tokenizer is stored in the checkpoint as 'vocab'
    print(generate_text(model, tokenizer, prompt=text, max_tokens=max_tokens, temperature=temperature))