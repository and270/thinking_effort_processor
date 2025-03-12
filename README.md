# Thinking Effort Controller

An experimental approach to controlling the reasoning depth of large language models that use explicit thinking tokens.

## Overview

This repository provides tools to dynamically adjust how much "thinking" a language model does during generation by manipulating the probability of the end-thinking token (`</think>`):

- **Low thinking effort (0.0)**: Model quickly exits the thinking phase
- **Normal thinking effort (1.0)**: No modification to the model's natural behavior
- **High thinking effort (>1.0)**: Model spends more time in the thinking phase

The `scale_factor` parameter controls the intensity of this effect:
- Higher values (e.g., 4) create a stronger contrast between low and high thinking effort
- The default value (2) works well for many models, but may need adjustment
- The actual scaling applied is calculated as: `scale = scale_factor ^ (1.0 - thinking_effort)`

This approach works with models trained with explicit reasoning patterns (using tokens like `<think>` and `</think>`), allowing control over reasoning depth without retraining.

## Installation

### For use with Transformers

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers
```

### For use with llama-cpp-python (GGUF quantized models)

```bash
pip install llama-cpp-python
```

(show how to reun the examples. For bauncing ball, we should do cd example/bouncing_ball   and then python inference_hight_thinking_llamacpp_py.py)

## How It Works

The controller scales the logits (prediction scores) for the end-thinking token based on the desired thinking effort:

- Scale = scale_factor ^ (1.0 - thinking_effort)
- When thinking_effort = 0, the end token is strongly boosted (less thinking)
- When thinking_effort = 1, no scaling occurs (normal thinking)
- When thinking_effort > 1, the end token is suppressed (more thinking)

Once the end-thinking token is generated, the controller stops modifying logits.

## Important Notes

- This is an experimental approach - results may vary across models
- You must identify the correct token ID for `</think>` in your specific model
- Different models may require different prompt formats and chat templates
- The `scale_factor` parameter may need adjustment based on the model

See the example Python files for implementation details and usage patterns.