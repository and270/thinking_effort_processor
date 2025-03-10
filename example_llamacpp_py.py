from llama_cpp import Llama
from thinking_effort_llamacpp_py import thinking_effort_processor


model_path = "C:/Users/andre/.cache/lm-studio/models/lmstudio-community/QwQ-32B-GGUF/QwQ-32B-Q6_K.gguf"

llm = Llama(model_path=model_path)

# Define the thinking effort level
thinking_effort = 0.0  # Very low thinking effort for this example
scale_factor = 4 #Controls the intensity of the scaling effect (default=2). For QwQ it seems we need bigger than 2.

# Get the token ID for the '</think>' token
end_thinking_token_id = 151668   #IMPORTANT: this is the token id for </think> ON QwQ model. If other model, YOU MUST check on tokenizer configs

processor = thinking_effort_processor(thinking_effort, end_thinking_token_id, scale_factor=scale_factor)
logits_processor = [processor]

#IMPORTANT: chat template for Qwen model. If other model, you must change the prompt format.
prompt = """<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
<think>
"""

# Stream the output with thinking effort
print("Streaming output with thinking effort:")
for chunk in llm.create_completion(
    prompt,
    max_tokens=8048,
    temperature=0.6,
    logits_processor=logits_processor,
    stream=True  # Enable streaming
):
    # Print the chunk text without newline to simulate streaming
    chunk_text = chunk['choices'][0]['text']
    print(chunk_text, end='', flush=True)

# Add a final newline
print()