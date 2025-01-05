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
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 48 # number of layers
    n_head: int = 25 # number of headsn_layer=48, n_head=25, n_embd=1600
    n_embd: int = 1600  # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

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
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

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
        j = 0  # Counter for decode iterations
        u = 0  # Counter for `err` and `err2`

        # Iterative processing for `N` steps
        for i in range(N):
            print(f"Iteration {i + 1}/{N}")
            for block in self.transformer.h:
                x = block(x, skip_mlp=skip_mlp)
                if const_reset:
                    self.apply(self._init_weights)
                y = self.transformer.ln_f(x)  # Normalize the output

                # Compute and update `err` and `err2`
                for k in range(y.shape[0]):
                    ya = y[k, :, :].squeeze()  # Extract the batch-specific output
                    y1 = torch.div(ya, torch.norm(ya, dim=1).unsqueeze(1))  # Normalize rows
                    aux = torch.abs((ya[0, :] / torch.norm(ya[0, :])) @ y1.T)  # Cosine similarity
                    aux2 = y1 @ y1.T  # Pairwise dot products

                    err2[0, :, :, u] = aux2  # Update err2
                    err[k, u, 0] = torch.mean(aux)  # Update mean cosine similarity
                    err[k, u, 1] = torch.mean(aux2)  # Update mean pairwise dot product
                u += 1  # Increment counter for `err` and `err2`

            # Save decoded outputs at specified iterations
            if decode_iters and i in decode_iters:
                err3[:, j, :, :] = self.lm_head(self.transformer.ln_f(x))
                j += 1

        x = self.transformer.ln_f(x)  # Final layer normalization
        logits = self.lm_head(x)  # Output logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, err, err2, err3
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

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
parser.add_argument("--use-model", type=int, nargs="?", help="Use the model normally to generate text. Provide max length as a positional value after flag.", const=50)
args = parser.parse_args()

# Handle input prompts
prompts = []
if args.random_prompt is None:
    if args.use_model is not None:
        prompt = input("Enter a prompt to generate text: ").strip()
        prompts = [prompt]
    else:
        while True:
            prompt = input(f"Enter prompt {len(prompts) + 1} (or press Enter to finish): ").strip()
            if not prompt:
                break
            prompts.append(prompt)

    if not prompts:
        prompts = [
            "Describe a futuristic city where humans and robots live together. Talk about what the city looks like and what daily life is like there."
        ]
model = GPT.from_pretrained('gpt2-xl')
model.eval()
model.to('cuda')

enc = tiktoken.get_encoding('gpt2')

if args.random_prompt:
    prompts = [' '.join([enc.decode([random.randint(0, 50257 - 2)]) for _ in range(args.random_prompt)])]

# Use model normally
if args.use_model is not None:
    max_length = args.use_model
    for prompt in prompts:
        tokens = enc.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to('cuda')
        while x.size(1) < max_length:
            with torch.no_grad():
                logits, _, _, _, _ = model(x,
                random_init=args.random_init,
                const_reset=args.const_reset,
                skip_mlp=args.noff,
                N=args.size,
            )
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1)

        # Decode and print the result
        tokens = x[0, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
else:
    # Forward pass and save outputs
    decode_iters = None
    if args.decode:
        decode_iters = list(map(int, args.decode.split(',')))

    results = []
    for i, prompt in enumerate(prompts):
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to('cuda')

        with torch.no_grad():
            logits, loss, err, err2, err3 = model(
                tokens,
                random_init=args.random_init,
                const_reset=args.const_reset,
                skip_mlp=args.noff,
                N=args.size,
                num_prompts=len(prompts),
                decode_iters=decode_iters,
            )

        if args.savefile:
            err_np = err.cpu().numpy()
            np.save(f"{args.savefile}.npy", err_np)

        if args.matrixerror:
            err2_np = err2.cpu().numpy()
            np.save(f"{args.matrixerror}.npy", err2_np)

        if decode_iters:
            decode_from_logits(err3, decode_iters, enc)
