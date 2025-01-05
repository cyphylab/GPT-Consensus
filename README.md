
# GPT-2 Model Script

This repository contains a Python script for a GPT-2 model implementation. The script allows for custom configurations, prompt inputs, and dynamic decoding based on specified iterations.

## Features

- **GPT-2 Model**: Implements a GPT-2 model with configurable parameters.
- **Dynamic Decoding**: Decode outputs at specific iterations using the `--decode` argument.
- **Customizable Prompts**: Accepts user-provided prompts, generates default prompts, or creates random prompts of a specified size.
- **Use Model Normally**: A special mode (`--use-model`) to generate text normally by providing a prompt. Use with `--size 1` to function normally.
- **Save Outputs**: Optionally save intermediate results (`err` and `err2`) to `.npy` files.

## Usage

Run the script with Python:
```bash
python GPT-2-Consensus.py [options]
```

### Arguments

| Argument           | Description                                                                 | Default            |
|---------------------|-----------------------------------------------------------------------------|--------------------|
| `--random-init`     | Applies random weight initialization once at the start of the forward pass.| `False`            |
| `--const-reset`     | Applies constant weight reset multiple times during the forward pass.      | `False`            |
| `--noff`            | Skips the MLP update in the forward pass.                                  | `False`            |
| `--size`            | Sets the value of the `N` variable for iterations.                         | `2000`             |
| `--matrixerror`     | Saves `err2` (matrix error) to a `.npy` file with the specified name.      | `None`             |
| `--savefile`        | Saves `err` (Eplot) to a `.npy` file with the specified name.              | `Eplot`            |
| `--decode`          | Specify iterations for decoding as a comma-separated list (e.g., `0,4,9`).| `None`             |
| `--random-prompt`   | Generates a completely random prompt of the specified size.                | `None`             |
| `--use-model`       | Use the model normally to generate text. Specify max length as a positional value after flag. | `50`            |

### Outputs: `err`, `err2`, and `err3`

- **`err`**:
  - Tracks layer-wise metrics for each prompt across iterations.
  - Stores:
    - Mean cosine similarity (`aux`) of token embeddings in the layer.
    - Mean pairwise dot products (`aux2`) of token embeddings.
  - Saved to the file specified by the `--savefile` argument (default: `Eplot.npy`).

- **`err2`**:
  - Tracks detailed token-to-token interactions for each prompt across iterations.
  - Stores pairwise dot products (`aux2`) between normalized token embeddings in the layer.
  - Saved to the file specified by the `--matrixerror` argument.

- **`err3`**:
  - Captures the model's output logits for specific iterations (if `--decode` is used).
  - Used to decode and display intermediate outputs.

### Examples

1. **Default Behavior**:
   ```bash
   python GPT-2-Consensus.py
   ```

2. **Generate a Random Prompt**:
   ```bash
   python GPT-2-Consensus.py --random-prompt 10
   ```

3. **Decode Specific Iterations**:
   ```bash
   python GPT-2-Consensus.py --decode 0,4,9
   ```

4. **Use Model Normally**:
   ```bash
   python GPT-2-Consensus.py --size 1 --use-model 100
   ```

   When prompted, enter the text you want the model to generate text for. This mode disables the `--decode`, `--matrixerror`, and `--savefile` arguments.

5. **Save Outputs**:
   ```bash
   python GPT-2-Consensus.py --matrixerror matrix_output --savefile error_output
   ```

---

## Requirements

- Python 3.7 or higher
- PyTorch
- NumPy
- tiktoken

Install dependencies:
```bash
pip install torch numpy tiktoken
```

---

## License

This project is licensed under the MIT License.
