import torch
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self, grammar):
        self.itos = {i:s for i,s in enumerate(grammar)}
        self.stoi = {s:i for i,s in enumerate(grammar)}

    def encode(self, content):
        return [self.stoi[x] for x in content]

    def decode(self, content):
        return ''.join([self.itos[x] for x in content])

class MLPDataset(Dataset):
    def __init__(self, text, tokenizer, T, stride):
        self.input_ids = []
        self.output_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - T, stride):
            self.input_ids.append(torch.tensor(token_ids[i: i+T]))
            self.output_ids.append(torch.tensor(token_ids[i+T]))

        self.input_ids = torch.stack(self.input_ids)
        self.output_ids = torch.stack(self.output_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.output_ids[index]

    def __len__(self):
        return len(self.input_ids)

def MLPDataLoader(text, tokenizer, B=4, T=8, stride=1, shuffle=True, drop_last=True, num_workers=0):

    dataset = MLPDataset(text, tokenizer, T, stride)

    dataloader = DataLoader(dataset, batch_size=B, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    return dataloader

def load_data(batch_size, context_length, stride, shuffle, drop_last, num_workers):
    with open('names.txt', 'r') as f:
        text = f.read()

    grammar = sorted(list(set(text)))

    tokenizer = Tokenizer(grammar)

    dataloader = MLPDataLoader(text, tokenizer, B=batch_size, T=context_length, stride=stride, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    print("Number of examples in the dataset:", len(dataloader.dataset))

    return grammar, tokenizer, dataloader