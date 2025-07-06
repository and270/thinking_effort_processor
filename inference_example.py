import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from thinking_effort_transformers import ThinkingEffortProcessor

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-1.7B"
THINKING_EFFORT = 1.3
SCALE_FACTOR = 2.0
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50

def find_end_thinking_token(tokenizer: AutoTokenizer) -> int:
    """
    Finds the token ID for the end-of-thinking marker for a given tokenizer.
    This is a crucial step as the token ID can vary between models.

    Args:
        tokenizer: The Hugging Face tokenizer instance.

    Returns:
        The integer token ID for the end-of-thinking marker.
    
    Raises:
        ValueError: If no candidate end-of-thinking token is found.
    """
    print("Attempting to find the end-of-thinking token ID...")
    # Common end-of-thinking tokens to check
    candidates = ["</think>", "</thinking>", "<|end_thinking|>"]
    
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        # Check if the candidate is a single token
        if len(token_ids) == 1:
            print(f"Found token '{candidate}' with ID: {token_ids[0]}")
            return token_ids[0]

    # Fallback for Qwen models which might have a specific known token ID.
    # This is an example, you might need to find the specific ID for your model.
    qwen_token_id = 151668 
    print(f"Could not find a standard end-of-thinking token. Using fallback ID for Qwen: {qwen_token_id}")
    print("NOTE: This fallback ID may not be correct for your model. Please verify.")
    return qwen_token_id

def find_keep_thinking_token(tokenizer: AutoTokenizer, token_str: str = "Wait") -> Optional[int]:
    """
    Finds the token ID for a "keep thinking" token, like "Wait".

    Args:
        tokenizer: The Hugging Face tokenizer instance.
        token_str: The string of the token to search for.

    Returns:
        The integer token ID if it's a single token, otherwise None.
    """
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(token_ids) == 1:
        print(f"Found 'keep thinking' token '{token_str}' with ID: {token_ids[0]}")
        return token_ids[0]
    
    print(f"Warning: '{token_str}' tokenized into multiple IDs: {token_ids}. Disabling keep-thinking logic.")
    return None


def run_inference():
    """
    Loads a model and runs a single inference example using the ThinkingEffortProcessor.
    """
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 1. Find the special token IDs for the chosen model
    end_thinking_token_id = find_end_thinking_token(tokenizer)
    keep_thinking_token_id = find_keep_thinking_token(tokenizer)

    # 2. Create the ThinkingEffortProcessor
    print(f"\nInitializing ThinkingEffortProcessor with effort={THINKING_EFFORT} and scale={SCALE_FACTOR}")
    logits_processor = ThinkingEffortProcessor(
        end_thinking_token_id=end_thinking_token_id,
        keep_thinking_token_id=keep_thinking_token_id,
        thinking_effort=THINKING_EFFORT,
        scale_factor=SCALE_FACTOR,
    )

    # 3. Prepare the prompt
    # This example uses the Qwen-3 chat template with thinking enabled.
    # You may need to adapt this for your specific model's template.
    prompt = "How much is 21+43*4/6?"
    messages = [
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # This is specific to Qwen's template
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 4. Generate the response
    print("\n--- Generating Response ---")
    print(f"Prompt:\n{prompt}")
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        logits_processor=[logits_processor],
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- Full Response ---")
    print(response)

if __name__ == "__main__":
    run_inference()
