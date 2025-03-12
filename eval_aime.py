import json
from datasets import load_dataset
from llama_cpp import Llama
from thinking_effort_llamacpp_py import thinking_effort_processor

# Load the AIME dataset
dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

# Model configuration
model_path = "C:/Users/andre/.cache/lm-studio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf"
end_thinking_token_id = 151649  # Token ID for '</think>' in DeepSeek-R1-Distill-Qwen-7B
scale_factor = 1.9  # Scale factor for thinking effort processor
thinking_effort = 2

# Initialize the model
print("Loading model...")
llm = Llama(model_path=model_path, n_ctx=131072)
print("Model loaded successfully!")

def run_with_thinking_effort(question, thinking_effort=thinking_effort):
    """Run inference with the thinking effort processor"""
    processor = thinking_effort_processor(thinking_effort, end_thinking_token_id, scale_factor=scale_factor)
    prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
    
    response = llm.create_completion(
        prompt,
        max_tokens=16384,
        temperature=0.6,
        logits_processor=[processor],
        stream=False
    )
    
    return response['choices'][0]['text']

def run_without_thinking_effort(question):
    """Run inference without the thinking effort processor"""
    prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
    
    response = llm.create_completion(
        prompt,
        max_tokens=16384,
        temperature=0.6,
        stream=False
    )
    
    return response['choices'][0]['text']

# Process dataset and save results
results = []
total_questions = len(dataset)

print(f"Processing {total_questions} questions...")

for i, item in enumerate(dataset):
    print(f"Processing question {i+1}/{total_questions}")
    
    problem = item["Problem"]
    gold_answer = item["Answer"]
    
    # Run with high thinking effort
    print("  Running with thinking effort...")
    response_with_thinking = run_with_thinking_effort(problem)
    
    # Run without thinking effort
    print("  Running without thinking effort...")
    response_without_thinking = run_without_thinking_effort(problem)
    
    # Save results
    result = {
        "problem": problem,
        "response_with_thinking": response_with_thinking,
        "response_without_thinking": response_without_thinking,
        "gold_answer": gold_answer,
        "thinking_effort": thinking_effort,
        "scale_factor": scale_factor
    }
    
    results.append(result)
    
    # Save intermediate results after each question
    with open("aime_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"  Results saved. Moving to next question...\n")

print("Evaluation complete! Results saved to aime_eval_results.json")
