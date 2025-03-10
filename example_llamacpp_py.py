from llama_cpp import Llama
from thinking_effort_llamacpp_py import thinking_effort_processor


model_path = "C:/Users/andre/.cache/lm-studio/models/lmstudio-community/QwQ-32B-GGUF/QwQ-32B-Q6_K.gguf"

llm = Llama(model_path=model_path)

# Define the thinking effort level
thinking_effort = 0.1  # Very low thinking effort for this example

# Get the token ID for the '</think>' token
end_thinking_token_id = 151668   #IMPORTANT: this is the token id for </think> ON QwQ model. If other model, YOU MUST check on tokenizer configs

processor = thinking_effort_processor(thinking_effort, end_thinking_token_id)
logits_processor = [processor]

#IMPORTANT: chat template for Qwen model. If other model, you must change the prompt format.
prompt = """<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
<think>
"""

# Regular inference if you want to compare
regular_output = llm.create_completion(
    prompt, 
    max_tokens=8048, 
    temperature=0.6
)
print("Regular Inference:", regular_output['choices'][0]['text'].strip())

# Thinking effort inference
thinking_output = llm.create_completion(
    prompt,
    max_tokens=8048,
    temperature=0.6,
    logits_processor=logits_processor
)

print("Thinking Effort Inference:", thinking_output['choices'][0]['text'].strip())