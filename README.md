
# GPT-2 Model Script

This repository contains a Python script for a GPT-2 model implementation. The script allows for custom configurations, prompt inputs, and dynamic decoding based on specified iterations.

## Features

- **GPT-2 Model**: Implements a GPT-2 model with configurable parameters.
- **Dynamic Decoding**: Decode outputs at specific iterations using the `--decode` argument.
- **Customizable Prompts**: Accepts user-provided prompts, generates default prompts, or creates random prompts of a specified size.
- **Save Outputs**: Optionally save intermediate results (`err` and `err2`) to `.npy` files.

## Usage

Run the script with Python:
```bash
python final_gpt_script_with_random_prompt.py [options]
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

### Examples

1. **Default Behavior**:
   ```bash
   python final_gpt_script_with_random_prompt.py
   ```

2. **Generate a Random Prompt**:
   ```bash
   python final_gpt_script_with_random_prompt.py --random-prompt 10
   ```

3. **Decode Specific Iterations**:
   ```bash
   python final_gpt_script_with_random_prompt.py --decode 0,4,9
   ```

4. **Save Outputs**:
   ```bash
   python final_gpt_script_with_random_prompt.py --matrixerror matrix_output --savefile error_output
   ```

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
