from llama_cpp import Llama
from thinking_effort_llamacpp_py import thinking_effort_processor

# Path to the downloaded GGUF model file
model_path = "path/to/DeepSeek-R1-Distill-Qwen-1.5B.gguf"

# Initialize the Llama model
llm = Llama(model_path=model_path)

# Define the thinking effort level
thinking_effort = 0.1  # Adjust between 0 and 2 as needed

# Get the token ID for the '</think>' token
end_thinking_token = "</think>"
end_thinking_token_id = llm.token_to_id(end_thinking_token)

# Create the logits processor function
logits_processor_fn = thinking_effort_processor(
    thinking_effort=thinking_effort,
    end_thinking_token_id=end_thinking_token_id
)

# Define the prompt
prompt = "What is the capital of France?"

# Generate text with regular inference (without the custom logits processor)
regular_output = llm.create_completion(
    prompt,
    max_tokens=8048,
    temperature=0.6
)

# Print the regular generated text
print("Regular Inference:", regular_output['choices'][0]['text'].strip())

# Generate text with the custom logits processor (thinking effort)
thinking_effort_output = llm.create_completion(
    prompt,
    max_tokens=8048,
    temperature=0.6,
    logits_processor=logits_processor_fn
)

# Print the thinking effort generated text
print("Thinking Effort Inference:", thinking_effort_output['choices'][0]['text'].strip())
