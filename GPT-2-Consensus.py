
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tiktoken
from dataclasses import dataclass
import random

# Self-attention mechanism for GPT-style models
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Linear layers for query, key, value
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # Output projection
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Split into query, key, value
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Compute scaled dot-product attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

# Feedforward network within each transformer block
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # Expand dimensionality
        self.gelu = nn.GELU(approximate='tanh')  # Activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # Project back

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Single transformer block: consists of self-attention and feedforward layers
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)  # Layer normalization before attention
        self.attn = CausalSelfAttention(config)  # Self-attention mechanism
        self.ln_2 = nn.LayerNorm(config.n_embd)  # Layer normalization before feedforward
        self.mlp = MLP(config)  # Feedforward network

    def forward(self, x, skip_mlp=False):
        x = x + self.attn(self.ln_1(x))  # Add residual connection
        if not skip_mlp:  # Optionally skip feedforward step
            x = x + self.mlp(self.ln_2(x))
        return x

# Configuration for the GPT model
@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum context size
    vocab_size: int = 50257  # Vocabulary size (GPT-2 specific)
    n_layer: int = 48  # Number of transformer blocks
    n_head: int = 25  # Number of attention heads
    n_embd = 1600  # Embedding dimensionality

# Main GPT model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Define transformer architecture
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Positional embeddings
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks
            ln_f=nn.LayerNorm(config.n_embd),  # Final layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Language model head
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        self.apply(self._init_weights)  # Initialize weights

    # Initialize weights for linear and embedding layers
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Forward pass through the model
    def forward(self, idx, targets=None, random_init=False, const_reset=False, skip_mlp=False, N=2000, num_prompts=1, decode_iters=None):
        B, T = idx.size()  # Batch size and sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # Embed tokens and positions
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        if random_init:
            self.apply(self._init_weights)  # Reinitialize weights

        # Initialize intermediate outputs
        err = torch.zeros((num_prompts, self.config.n_layer * N, 2), device=idx.device)
        err2 = torch.zeros((num_prompts, T, T, self.config.n_layer * N), device=idx.device)
        err3 = torch.zeros((num_prompts, len(decode_iters) if decode_iters else 0, T, 50257), device=idx.device)
        j = 0

        # Iterative processing for `N` steps
        for i in range(N):
            for block in self.transformer.h:
                x = block(x, skip_mlp=skip_mlp)
                if const_reset:
                    self.apply(self._init_weights)

            if decode_iters and i in decode_iters:
                err3[:, j, :, :] = self.lm_head(self.transformer.ln_f(x))
                j += 1

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, err, err2, err3

# Decode outputs from saved logits
def decode_from_logits(err3, decode_iters, enc):
    num_prompts, num_decoded_inputs, size_of_prompt, _ = err3.shape

    for j in range(num_decoded_inputs):
        for prompt_idx in range(num_prompts):
            logits = err3[prompt_idx, j, :, :]
            decoded_tokens = logits.argmax(dim=-1).tolist()
            decoded_output = enc.decode(decoded_tokens)
            print(f"Decoded output for prompt {prompt_idx + 1}, iteration {decode_iters[j]}:")
            print(f"> {decoded_output}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run GPT model with optional configurations.")
parser.add_argument("--random-init", action="store_true", help="Reinitialize weights at the start.")
parser.add_argument("--const-reset", action="store_true", help="Reinitialize weights multiple times during processing.")
parser.add_argument("--noff", action="store_true", help="Skip the feedforward layer updates.")
parser.add_argument("--size", type=int, default=2000, help="Set the number of iterations.")
parser.add_argument("--matrixerror", type=str, help="Save matrix error (err2) to a file.")
parser.add_argument("--savefile", type=str, default="Eplot", help="File to save err (Eplot).")
parser.add_argument("--decode", type=str, help="Specify iterations for decoding (comma-separated).")
parser.add_argument("--random-prompt", type=int, help="Generate a random prompt of the specified length.")
args = parser.parse_args()

config = GPTConfig()
model = GPT(config)
model.eval()
model.to('cuda')

enc = tiktoken.get_encoding('gpt2')

# Handle input prompts
prompts = []
if args.random_prompt:
    prompts = [' '.join([enc.decode([random.randint(0, config.vocab_size - 1)]) for _ in range(args.random_prompt)])]
else:
    while True:
        prompt = input(f"Enter prompt {len(prompts) + 1} (or press Enter to finish): ").strip()
        if not prompt:
            break
        prompts.append(prompt)

if not prompts:
    prompts = [
        "After endless years lost in the shadows of Shakespeare's sonnets and the melancholic musings of Pessoa, I have glimpsed enlightenment's elusive light. Now, on the precipice of my final hour, as the weight of mortality presses upon me, I must reveal to you the one truth that transcends all others—the meaning of life is…"
    ]

# Process decoding iterations
decode_iters = None
if args.decode:
    decode_iters = list(map(int, args.decode.split(',')))

# Forward pass and save outputs
results = []
for i, prompt in enumerate(prompts):
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    x = tokens.to('cuda')

    with torch.no_grad():
        logits, loss, err, err2, err3 = model(
            x,
            random_init=args.random_init,
            const_reset=args.const_reset,
            skip_mlp=args.noff,
            N=args.size,
            num_prompts=len(prompts),
            decode_iters=decode_iters,
        )

    err_np = err.cpu().numpy()
    np.save(f"{args.savefile}.npy", err_np)

    if args.matrixerror:
        err2_np = err2.cpu().numpy()
        np.save(f"{args.matrixerror}.npy", err2_np)

    if decode_iters:
        decode_from_logits(err3, decode_iters, enc)
