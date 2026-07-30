# microGPT

A character-level GPT built from scratch with PyTorch.

The project retains the core ideas behind the decoder-only Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), but instead of Post-LN, it implements the Pre-LN Transformer variant ([Xiong et al., 2020](https://arxiv.org/pdf/2002.04745)).

This implementation covers only the pre-training phase and doesn't cover the fine-tuning phase.

## Getting started

You need Python and PyTorch installed.

```bash
git clone https://github.com/shanthan5589/microGPT.git
cd microGPT
python -m pip install torch
```

Train the model:

```bash
python train.py
```

Training uses CUDA when it is available, otherwise it runs on the CPU. When it finishes, it writes a `model.pt` checkpoint to the project directory.

Generate text from the trained model:

```bash
python sample.py
```

You can change the prompt, max output tokens, and temperature in `sample.py`:

```python
text = " "
max_tokens = 100
temperature = 1.0
```

<!-- ## How it works

The tokenizer maps every unique character in `input.txt` to an integer. The dataset turns the resulting sequence into fixed-length input and target windows, where each target is the input shifted forward by one character.

The model learns to predict the next character using:

- learned token and position embeddings
- masked multi-head self-attention
- feed-forward layers with residual connections
- a final linear layer over the character vocabulary

During generation, the model repeatedly samples one character from its predicted probability distribution and appends it to the prompt. -->

## Project structure

| File | Purpose |
| --- | --- |
| `model.py` | Transformer model |
| `data.py` | Character tokenizer, dataset, and data loader |
| `train.py` | Training loop and checkpoint creation |
| `sample.py` | Text generation from a trained checkpoint |
| `input.txt` | Training corpus |

## Configuration

Model and training settings are kept at the top of `train.py`.

```python
context_length = 8
embed_dim = 32
n_layers = 4
num_heads = 4

epochs = 1
learning_rate = 1e-3
batch_size = 4
```

To train on your own text, replace contents of `input.txt` with your text and run `train.py` again. Prompts passed to the sampler may only contain characters that appeared in the training corpus.