#example usage
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from thinking_effort_transformers import ThinkingEffortProcessor

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()

think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")

# Create the custom processor
thinking_effort_processor = ThinkingEffortProcessor(
    end_thinking_token_id=think_end_token_id,
    thinking_effort=0.1 #very low thinking effort
)

logits_processor = LogitsProcessorList([thinking_effort_processor])

messages = [{"role": "user", "content": "What is the capital of France?"}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)


# Compare to regular inference
regular_output = model.generate(
    input_ids,
    max_new_tokens=8048,
    do_sample=True,
    temperature=0.6,
)
regular_response = tokenizer.decode(regular_output[0][input_ids.shape[-1]:], skip_special_tokens=True)
print("Regular Inference:", regular_response.strip())


# Thinking Effort Inference
thinking_effort_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=8048,
    logits_processor=logits_processor,
    do_sample=True,
    temperature=0.6,
)

thinking_effort_response = tokenizer.decode(
    thinking_effort_output[0][input_ids.shape[-1]:], skip_special_tokens=True
)

print("Thinking Effort Inference:", thinking_effort_response.strip())
