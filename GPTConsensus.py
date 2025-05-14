# GPT‑Neo 2.7B Consensus / Generation (model‑parallel)
# ---------------------------------------------------
# Flags:
#   --size N          : number of consensus iterations (default 1000)
#   --noff            : skip feed‑forward (MLP) blocks in each layer
#   --randominit      : randomly initialize model weights before running
#   --randomprompt L  : generate a random L‑token prompt
#   --use-model K     : generate K tokens via consensus sampling
#   --savefile FILE   : save hidden state and diagnostics (consensus) or generated text and diagnostics

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel, AutoTokenizer
from torch.nn import functional as F

# ---------------- initialize random seeds ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- helper: skip MLP ----------------
def patch_skip_mlp(model):
    """
    Remove feed‑forward sub‑block: bypass the second LayerNorm and zero out the MLP.
    """
    class ZeroMLP(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return torch.zeros_like(hidden_states)
    for block in model.transformer.h:
        block.ln_2 = nn.Identity()
        block.mlp  = ZeroMLP()

# ---------------- consensus helper ----------------
def consensus(model, seq, loop_size: int, skip_mlp: bool):
    """
    Run GPT‑Neo transformer loop_size times on seq of token IDs.
    Returns (hidden, diag) where diag has shape (B, loop_size, L)
    of mean cosine similarity of token 0 vs. all tokens after each layer.
    """
    if skip_mlp:
        patch_skip_mlp(model)
    # initial embedding + positional
    B, T = seq.shape
    wte = model.transformer.wte
    wpe = model.transformer.wpe
    pos_ids = torch.arange(T, device=seq.device).unsqueeze(0)
    hidden = wte(seq) + wpe(pos_ids)

    # number of transformer blocks
    L = len(model.transformer.h)
    # diagnostic trace per iteration and per layer
    diag = torch.zeros((B, loop_size *L+1), dtype=torch.float32, device=seq.device)
    # iterative consensus passes
    for batch_idx in range(B):
                # Extract the hidden states for a single prompt (tokens in that prompt)        
                hn = (hidden[batch_idx])
                ya = hn.squeeze()
                y1 = torch.div(ya, torch.norm(ya, dim=1).unsqueeze(1))  # Normalize rows
                aux = torch.abs((ya[0, :] / torch.norm(ya[0, :])) @ y1.T)
                diag[batch_idx, 0] = 1-torch.mean(aux)
    u=1
    for i in range(loop_size):
        print(f"consensus step {i+1}/{loop_size}")
        # pass through each layer, record metric
        for layer_idx, block in enumerate(model.transformer.h):
            outputs = block(hidden)
            hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            for batch_idx in range(B):
                # Extract the hidden states for a single prompt (tokens in that prompt)        
                hn = (hidden[batch_idx])
                ya = hn.squeeze()
                y1 = torch.div(ya, torch.norm(ya, dim=1).unsqueeze(1))  # Normalize rows
                aux = torch.abs((ya[0, :] / torch.norm(ya[0, :])) @ y1.T)
                diag[batch_idx, u] = torch.mean(aux)
            u=u+1
            
        hidden = model.transformer.ln_f(hidden)
        
    return hidden, diag

# ---------------- sampling helper ----------------
def sample_top_p(logits, p: float = 0.8):
    """Nucleus (top‑p) sampling"""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumprobs > p
    mask[..., :1] = False
    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum(-1, keepdim=True)
    local_idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx.gather(-1, local_idx)

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",         type=int, help="consensus iterations N", default=1000)
    parser.add_argument("--noff",         action="store_true", help="skip MLP blocks")
    parser.add_argument("--randominit",   action="store_true", help="randomly initialize model weights before running")
    parser.add_argument("--randomprompt", type=int, nargs=2, metavar=("N","L"), help="generate N random prompts of length L tokens")
    parser.add_argument("--use-model",    type=int, nargs="?", const=50,
                        help="generate this many tokens via consensus sampling (default=50)")
    parser.add_argument("--savefile",     type=str, help="prefix for .npy outputs")
    parser.add_argument("--nneo",         action="store_true", help="use GPT-2-XL (gpt2-xl) instead of GPT-Neo-2.7B")
    args = parser.parse_args()

    if args.nneo:
        model_name = "gpt2-xl"
        ModelClass = GPT2LMHeadModel
    else:
        model_name = "EleutherAI/gpt-neo-2.7B"
        ModelClass = GPTNeoForCausalLM

# load tokenizer & model
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token

    model = ModelClass.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # optionally randomize weights
    if args.randominit:
        print("Randomly initializing model weights...")
        def reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        model.apply(reset)

    model.eval()

    # reference device for input_ids
    ref_dev = next(model.parameters()).device

    # build prompt
    if args.randomprompt:
        N, L = args.randomprompt
        rand = [
            [random.randint(0, tok.vocab_size - 1) for _ in range(L)]
            for _ in range(N)
        ]
        prompt_ids = torch.tensor(rand, device=ref_dev)
        if N==1 :
            prompt_ids = prompt_ids.unsqueeze() 
    else:
        text = input("Prompt: ").strip() or "Describe a futuristic city where humans and robots live together. Talk about what the city looks like and what daily life is like there."
        prompt_ids = tok(text, return_tensors="pt").input_ids.to(ref_dev)

    # generation via consensus sampling
    if args.use_model is not None:
        K = args.use_model
        seq = prompt_ids.clone()
        all_diag = []
        with torch.no_grad():
            for step in range(K):
                print(f"generation step {step+1}/{K}")
                hidden, diag = consensus(
                    model, seq, loop_size=args.size, skip_mlp=args.noff
                )
                all_diag.append(diag.cpu().numpy())

                # sample next token from last hidden
                logits_last = model.lm_head(hidden[:, -1])
                next_id = sample_top_p(logits_last, p=0.8).to(seq.device)
                seq = torch.cat([seq, next_id], dim=1)

        text_out = tok.decode(seq[0], skip_special_tokens=True)
        print("\n> ", text_out)
        if args.savefile:
            np.save(args.savefile + ".npy", np.array([text_out]))
            np.save(args.savefile + "_diag.npy",  np.vstack(all_diag))
        return

    # consensus-only
    with torch.no_grad():
        hidden, diag = consensus(
            model, prompt_ids, loop_size=args.size, skip_mlp=args.noff
        )
    if args.savefile:
        np.save(args.savefile + "_hidden.npy", hidden.cpu().float().numpy())
        np.save(args.savefile + ".npy",    diag.cpu().numpy())
    print("Finished consensus.")

if __name__ == "__main__":
    main()
