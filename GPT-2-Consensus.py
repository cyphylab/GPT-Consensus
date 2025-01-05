
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tiktoken
from dataclasses import dataclass
import random


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, skip_mlp=False):
        x = x + self.attn(self.ln_1(x))
        if not skip_mlp:
            x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 48
    n_head: int = 25
    n_embd = 1600


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, random_init=False, const_reset=False, skip_mlp=False, N=2000, num_prompts=1, decode_iters=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        if random_init:
            self.apply(self._init_weights)

        err = torch.zeros((num_prompts, self.config.n_layer * N, 2), device=idx.device)
        err2 = torch.zeros((num_prompts, T, T, self.config.n_layer * N), device=idx.device)
        err3 = torch.zeros((num_prompts, len(decode_iters) if decode_iters else 0, T, 50257), device=idx.device)
        j = 0

        for i in range(N):
            print(f"Iteration {i + 1}/{N}")
            for block in self.transformer.h:
                x = block(x, skip_mlp=skip_mlp)
                if const_reset:
                    self.apply(self._init_weights)

            if decode_iters and i in decode_iters:
                print(f"Saving logits for decode iteration {i}")
                err3[:, j, :, :] = self.lm_head(self.transformer.ln_f(x))
                j += 1

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, err, err2, err3


def decode_from_logits(err3, decode_iters, enc):
    num_prompts, num_decoded_inputs, size_of_prompt, _ = err3.shape

    for j in range(num_decoded_inputs):
        for prompt_idx in range(num_prompts):
            logits = err3[prompt_idx, j, :, :]
            decoded_tokens = logits.argmax(dim=-1).tolist()
            decoded_output = enc.decode(decoded_tokens)
            print(f"Decoded output for prompt {prompt_idx + 1}, iteration {decode_iters[j]}:")
            print(f"> {decoded_output}")


parser = argparse.ArgumentParser(description="Run GPT model with optional configurations.")
parser.add_argument("--random-init", action="store_true", help="Apply random initialization once at the start of the forward pass.")
parser.add_argument("--const-reset", action="store_true", help="Apply constant weight reset multiple times in the forward pass.")
parser.add_argument("--noff", action="store_true", help="Skip MLP update in the forward pass.")
parser.add_argument("--size", type=int, default=2000, help="Set the value of the N variable (default: 2000).")
parser.add_argument("--matrixerror", type=str, help="Save matrix error (err2) to a file with the specified name.")
parser.add_argument("--savefile", type=str, default="Eplot", help="Save err (err1) to a file with the specified name (default: Eplot).")
parser.add_argument("--decode", type=str, help="Specify iterations for decoding as a comma-separated list (e.g., 0,4,9).")
parser.add_argument("--random-prompt", type=int, help="Generate a completely random prompt of the specified size.")
args = parser.parse_args()

config = GPTConfig()
model = GPT(config)
model.eval()
model.to('cuda')

enc = tiktoken.get_encoding('gpt2')

prompts = []
if args.random_prompt:
    print(f"Generating a random prompt of size {args.random_prompt}.")
    prompts = [' '.join([enc.decode([random.randint(0, config.vocab_size - 1)]) for _ in range(args.random_prompt)])]
else:
    print("You can input prompts. Leave empty and press Enter to stop.")
    while True:
        prompt = input(f"Enter prompt {len(prompts) + 1} (or press Enter to finish): ").strip()
        if not prompt:
            break
        prompts.append(prompt)

if not prompts:
    print("No prompts provided. Using a single default prompt.")
    prompts = [
        "After endless years lost in the shadows of Shakespeare's sonnets and the melancholic musings of Pessoa, I have glimpsed enlightenment's elusive light. Now, on the precipice of my final hour, as the weight of mortality presses upon me, I must reveal to you the one truth that transcends all others—the meaning of life is…"
    ]

decode_iters = None
if args.decode:
    decode_iters = list(map(int, args.decode.split(',')))
    print(f"Decoding at iterations: {decode_iters}")

results = []
for i, prompt in enumerate(prompts):
    print(f"Processing Prompt {i + 1}/{len(prompts)}: {prompt}")
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
    print(f"Saved err (Eplot) to {args.savefile}.npy")

    if args.matrixerror:
        err2_np = err2.cpu().numpy()
        np.save(f"{args.matrixerror}.npy", err2_np)
        print(f"Saved matrix error (err2) to {args.matrixerror}.npy")

    if decode_iters:
        decode_from_logits(err3, decode_iters, enc)

print("Processing complete.")
