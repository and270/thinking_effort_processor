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

To use the `ThinkingEffortProcessor` with Hugging Face Transformers, install the following libraries:

```bash
pip install torch transformers accelerate
```

## How It Works

The controller scales the logits (prediction scores) for the end-thinking token based on the desired thinking effort:

- `scale = scale_factor ^ (1.0 - thinking_effort)`
- When `thinking_effort = 0`, the end token is strongly boosted (less thinking)
- When `thinking_effort = 1`, no scaling occurs (normal thinking)
- When `thinking_effort > 1`, the end token is suppressed (more thinking)

The logic is implemented in the `ThinkingEffortProcessor` class, which is a `LogitsProcessor` for the `transformers` library. Once the end-thinking token is generated for a sequence in a batch, the controller stops modifying the logits for that sequence.

## Important Notes

- This is an experimental approach—results may vary across models.
- You must identify the correct token ID for `</think>` in your specific model. The evaluation script provides examples of how to do this.
- Different models may require different prompt formats and chat templates.
- The `scale_factor` parameter may need adjustment based on the model.

## Running the Evaluation

This repository includes a script to evaluate the `ThinkingEffortProcessor` on the `gsm8k` benchmark. It tests various combinations of `thinking_effort` and `scale_factor` and saves the results to a CSV file.

### 1. Install Dependencies

To run the evaluation script, you will need to install a few additional packages:

```bash
pip install transformers datasets pandas accelerate
```

### 2. Configure the Evaluation

Open the `eval/evaluate_thinking_effort.py` file and modify the `EVALUATION_CONFIG` dictionary at the top to set up your test run.

Key parameters include:
- `model_name`: The Hugging Face model you want to evaluate.
- `thinking_efforts`: A list of thinking effort values to test.
- `scale_factors`: A list of scale factors to test.
- `max_questions`: The number of questions from the `gsm8k` test set to use.

### 3. Run the Script

Execute the script from the root directory of the project:

```bash
python eval/evaluate_gsm8k.py
```

The script will print its progress to the console and save a detailed `gsm8k_thinking_effort_results_{timestamp}.csv` file in the root directory upon completion.

## Running the AIME 2025 Evaluation

This repository also includes a script to evaluate the `ThinkingEffortProcessor` on the `AIME/AIME` benchmark (2025-I subset). It follows the same principles as the GSM8k evaluator.

### 1. Install Dependencies

Ensure you have the necessary packages installed:

```bash
pip install transformers datasets pandas accelerate
```

### 2. Configure the Evaluation

Open the `eval/evaluate_aime2025.py` file and modify the `EVALUATION_CONFIG` dictionary. The key parameters are similar to the GSM8k evaluation, but are tailored for the AIME dataset:
- `model_name`: The Hugging Face model to evaluate.
- `dataset_name`: The dataset name on Hugging Face (default: `"AIME/AIME"`).
- `dataset_config`: The specific AIME competition to use (default: `"2025-I"`).
- `thinking_efforts`: A list of thinking effort values.
- `scale_factors`: A list of scale factors.
- `max_questions`: The number of questions to test.

### 3. Run the Script

Execute the script from the root directory of the project:

```bash
python eval/evaluate_aime2025.py
```

The script will save its results to `aime2025_thinking_effort_results_{timestamp}.csv`.

## Running a Single Inference Example

The `inference_example.py` script provides a clear, minimal example of how to use the `ThinkingEffortProcessor` for a single generation task. It is a good starting point for integrating the processor into your own code.

### How it Works

The script is configured with three presets for thinking effort:
-   `"low"`: `thinking_effort=0.7`, `scale_factor=2.0`
-   `"medium"`: `thinking_effort=1.0`, `scale_factor=2.0`
-   `"high"`: `thinking_effort=1.3`, `scale_factor=2.0`

It demonstrates several key steps:
1.  **Finding Special Token IDs**: It includes a function `find_end_thinking_token` that shows how to programmatically find the crucial `</think>` token ID for your model. This is a necessary first step.
2.  **Processor Initialization**: It shows how to create an instance of `ThinkingEffortProcessor` based on the selected preset.
3.  **Model Generation**: It runs a single prompt through the model with the processor enabled to control the thinking process.

### Run the Example

To run the example, execute the following command from the root of the repository:

```bash
python inference_example.py
```

By default, the script runs with the `"high"` preset. You can easily change this by modifying the last line in `inference_example.py` to use `"medium"` or `"low"` to experiment with different settings.