import torch
from model import GPT
from train import tokenizer

text = ""
max_tokens = 100
temperature = 1.0

def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0):
    model.eval()
    idx = tokenizer.encode(prompt)
    output = model.generate(idx, max_tokens=max_tokens, temperature=temperature)
    return output

def __init__():
    ckpt = torch.load('model.pt')
    model = GPT(**ckpt['model_args'])
    model.load_state_dict(ckpt['state_dict'])
    print(generate_text(model, tokenizer, prompt=text, max_tokens=max_tokens, temperature=temperature))