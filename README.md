# GPTNeo / GPT‑2-XL Consensus & Generation

This repository contains a script for consensus-based decoding and generation using either **GPT-Neo-2.7B** or **GPT-2-XL**, with additional functionality for skipping MLPs, generating random prompts, and saving similarity diagnostics.

## Features

- **Model Choice**: Use `EleutherAI/gpt-neo-2.7B` by default or switch to `gpt2-xl` using `--nneo`.
- **Prompt Sampling**: Generate a batch of random prompts of arbitrary size and token length.
- **Skip MLPs**: Remove the feedforward sublayers (`--noff`) to study their influence.
- **Random Initialization**: Reset model weights with `--randominit` to test stability.
- **Text Generation**: Generate output using consensus-informed token sampling (`--use-model`).
- **Save Outputs**: Export hidden state and similarity metrics to `.npy` files.
-  **Random Initialization**: Reset model weights with `--randominit` or `--fullrandom` to test stability.

## Usage

```bash
python GPTConsensus.py [options]
```

### Arguments

| Argument               | Description                                                                   | Default     |
|------------------------|-------------------------------------------------------------------------------|-------------|
| `--size N`             | Number of consensus iterations (forward passes per token).                    | `1000`      |
| `--noff`               | Skip MLP blocks inside each transformer layer.                                | `False`     |
| `--randominit`         | Re-initialize model weights randomly at startup.                              | `False`     |
| `--randomprompt N L`   | Generate `N` prompts, each with `L` randomly sampled tokens.                  | `None`      |
| `--use-model K`        | Generate `K` tokens with consensus-based top-p sampling.                      | `None`      |
| `--savefile NAME`      | Prefix for `.npy` files to save hidden states or metrics.                     | `None`      |
| `--nneo`               | Use GPT-2-XL (`gpt2-xl`) instead of GPT-Neo (`EleutherAI/gpt-neo-2.7B`).      | `False`     |
| `--fullrandom`         | Re-initialize model weights randomly at startup and after each model pass.    | `False`     |

## Output Files

When `--savefile prefix` is set:

- `prefix.npy` – diagnostic matrix tracking similarity over layers and steps.
- `prefix_hidden.npy` – final hidden states from consensus (if no generation).
- `prefix.npy` (overwritten) – generated text (if `--use-model` is active).
- `prefix_diag.npy` – stack of per-token diagnostic traces during generation.

## Examples

1. **Run 500 consensus iterations on a user-defined prompt**:
   ```bash
   python GPTConsensus.py --size 500
   ```

2. **Use GPT-2-XL and skip MLPs**:
   ```bash
   python GPTConsensus.py --nneo --noff
   ```

3. **Run consensus on 5 random prompts of length 50**:
   ```bash
   python GPTConsensus.py --randomprompt 5 50 --size 300
   ```

4. **Generate 100 tokens using consensus-enhanced sampling**:
   ```bash
   python GPTConsensus.py --use-model 100 --size 20
   ```

5. **Save output diagnostics**:
   ```bash
   python GPTConsensus.py --randomprompt 1 100 --savefile output_metrics
   ```

## Requirements

- Python 3.8+
- PyTorch with GPU support
- NumPy
- Transformers

Install with pip:
```bash
pip install torch numpy transformers
```

## License

MIT License
