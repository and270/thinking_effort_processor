#!/usr/bin/env python3
"""
Evaluation script for ThinkingEffortProcessor on the AIME 2025 benchmark.
Tests various thinking_effort and scale_factor configurations and saves detailed results to CSV.
"""

import re
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import itertools
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd

from thinking_effort_transformers import ThinkingEffortProcessor

EVALUATION_CONFIG = {
    "model_name": "Qwen/Qwen3-1.7B",
    "dataset_name": "opencompass/AIME2025",
    "dataset_config": "AIME2025-I",
    "dataset_split": "test",
    "thinking_efforts": [1.1, 1.3, 1.4],
    "scale_factors": [2.0, 2.5, 3.0],
    "max_questions": 15,
    "max_new_tokens": 32768,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "include_baseline": True,
}


class AIME2025Evaluator:
    def __init__(self, model_name: str, dataset_name: str, dataset_config: str, dataset_split: str, 
                 max_new_tokens: int, temperature: float, top_p: float, top_k: int, device: str = "auto"):
        """
        Initialize the evaluator with a model and tokenizer.
        
        Args:
            model_name: HuggingFace model name
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration (e.g., '2025-I')
            dataset_split: Dataset split to use (e.g., 'test')
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature for generation
            top_p: Top-p for nucleus sampling
            top_k: Top-k for sampling
            device: Device to use for inference ("auto", "cuda", "cpu")
        """
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True
        )
        
        # Find the end-of-thinking token ID
        self.end_thinking_token_id = self.find_end_thinking_token()
        print(f"End thinking token ID: {self.end_thinking_token_id}")

        # Find the keep-thinking token ID for "Wait"
        wait_token_ids = self.tokenizer.encode("Wait", add_special_tokens=False)
        if len(wait_token_ids) == 1:
            self.keep_thinking_token_id = wait_token_ids[0]
            print(f"Keep thinking token 'Wait' ID: {self.keep_thinking_token_id}")
        else:
            print(f"Warning: 'Wait' tokenized into {wait_token_ids}. Disabling keep-thinking logic.")
            self.keep_thinking_token_id = None
        
        # Load AIME dataset
        print(f"Loading dataset: {dataset_name}, config: {dataset_config}, split: {dataset_split}")
        self.dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        print(f"Loaded {len(self.dataset)} test examples")
        
        # Results storage
        self.results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_filename = f"aime2025_thinking_effort_results_{timestamp}.csv"
        print(f"Results will be saved incrementally to: {self.results_filename}")
        
    def find_end_thinking_token(self) -> int:
        """Find the token ID for the end-of-thinking marker."""
        # Common end-of-thinking tokens
        candidates = ["</think>", "</thinking>", "<|end_thinking|>"]
        
        for candidate in candidates:
            token_id = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_id) == 1:
                return token_id[0]
        
        # Fallback for Qwen model (known token ID)
        return 151668
    
    def extract_answer(self, text: str) -> Optional[int]:
        """Extract the final answer from the model's response."""
        # The required format is \boxed{{ANSWER}}
        match = re.search(r'\\boxed\{\{([0-9]+)\}\}', text)
        if match:
            return int(match.group(1))

        # Fallback to find the last number if the format is not exact
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[-1])
        return None
    
    def extract_gold_answer(self, answer_text: str) -> int:
        """Extract the gold answer from AIME dataset."""
        return int(answer_text)
    
    def count_thinking_tokens(self, text: str) -> Tuple[int, int]:
        """
        Count thinking tokens and total tokens in the response.
        
        Returns:
            Tuple of (thinking_tokens, total_tokens)
        """
        total_tokens = len(self.tokenizer.encode(text))
        
        # Find thinking sections (between <think> and </think> or similar)
        thinking_patterns = [
            r"<think>(.*?)</think>",
            r"<thinking>(.*?)</thinking>",
            r"<|thinking\|>(.*?)<\|end_thinking\|>",
        ]
        
        thinking_tokens = 0
        for pattern in thinking_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                thinking_tokens += len(self.tokenizer.encode(match))
        
        return thinking_tokens, total_tokens
    
    def generate_response(self, question: str, thinking_effort: Optional[float], scale_factor: Optional[float]) -> Dict:
        """
        Generate a response to a question, optionally using the thinking effort processor.
        
        Returns:
            Dictionary with response details
        """
        # Create the thinking effort processor only if parameters are provided
        logits_processor = []
        if thinking_effort is not None and scale_factor is not None:
            processor = ThinkingEffortProcessor(
                end_thinking_token_id=self.end_thinking_token_id,
                keep_thinking_token_id=self.keep_thinking_token_id,
                thinking_effort=thinking_effort,
                scale_factor=scale_factor
            )
            logits_processor.append(processor)
        
        # Format the prompt as per user specification
        prompt_text = f"{question}. Put your final answer within \\boxed{{}}.The answer is an integer between 0 and 999 inclusive."
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Keep thinking tokens if model supports it
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                logits_processor=logits_processor if logits_processor else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=False)
        
        # Count tokens
        thinking_tokens, total_tokens = self.count_thinking_tokens(response)
        
        # Extract answer
        predicted_answer = self.extract_answer(response)
        
        return {
            "response": response,
            "predicted_answer": predicted_answer,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "generation_time": generation_time,
        }
    
    def evaluate_single_question(self, question: str, gold_answer: int, 
                               thinking_effort: Optional[float], scale_factor: Optional[float], 
                               question_idx: int) -> Dict:
        """Evaluate a single question with given parameters."""
        if thinking_effort is None:
            print(f"Evaluating question {question_idx + 1} (Baseline)")
        else:
            print(f"Evaluating question {question_idx + 1} with effort={thinking_effort}, scale={scale_factor}")
        
        try:
            result = self.generate_response(question, thinking_effort, scale_factor)
            
            # Check correctness for integer answers
            is_correct = False
            if result["predicted_answer"] is not None:
                is_correct = result["predicted_answer"] == gold_answer
            
            return {
                "question_idx": question_idx,
                "question": question,
                "gold_answer": gold_answer,
                "thinking_effort": "baseline" if thinking_effort is None else thinking_effort,
                "scale_factor": "baseline" if scale_factor is None else scale_factor,
                "predicted_answer": result["predicted_answer"],
                "is_correct": is_correct,
                "thinking_tokens": result["thinking_tokens"],
                "total_tokens": result["total_tokens"],
                "generation_time": result["generation_time"],
                "response": result["response"],
            }
            
        except Exception as e:
            print(f"Error evaluating question {question_idx}: {e}")
            return {
                "question_idx": question_idx,
                "question": question,
                "gold_answer": gold_answer,
                "thinking_effort": "baseline" if thinking_effort is None else thinking_effort,
                "scale_factor": "baseline" if scale_factor is None else scale_factor,
                "predicted_answer": None,
                "is_correct": False,
                "thinking_tokens": 0,
                "total_tokens": 0,
                "generation_time": 0.0,
                "response": f"ERROR: {str(e)}",
            }
    
    def save_result(self, result: Dict):
        """Append a single result to the CSV file."""
        df = pd.DataFrame([result])
        file_exists = os.path.exists(self.results_filename)
        df.to_csv(self.results_filename, mode='a', header=not file_exists, index=False)
    
    def evaluate_configurations(self, 
                              thinking_efforts: List[float],
                              scale_factors: List[float],
                              max_questions: int = 15,
                              include_baseline: bool = True) -> None:
        """
        Evaluate all combinations of thinking_effort and scale_factor parameters.
        
        Args:
            thinking_efforts: List of thinking effort values to test
            scale_factors: List of scale factor values to test
            max_questions: Maximum number of questions to evaluate (for faster testing)
            include_baseline: Whether to include a baseline run without the processor.
        """
        # Limit dataset size for testing
        test_dataset = self.dataset.select(range(min(max_questions, len(self.dataset))))
        
        # Generate all parameter combinations
        config_combinations = list(itertools.product(thinking_efforts, scale_factors))
        
        if include_baseline:
            config_combinations.insert(0, (None, None))
        
        print(f"Testing {len(config_combinations)} configurations on {len(test_dataset)} questions")
        
        for config_idx, (thinking_effort, scale_factor) in enumerate(config_combinations):
            print(f"\n--- Configuration {config_idx + 1}/{len(config_combinations)} ---")
            if thinking_effort is None:
                print("Testing Baseline (no thinking effort processor)")
            else:
                print(f"Thinking effort: {thinking_effort}, Scale factor: {scale_factor}")
            
            config_results = []
            running_correct_count = 0
            running_total_tokens = 0
            
            for question_idx, example in enumerate(test_dataset):
                question = example["question"]
                gold_answer = self.extract_gold_answer(example["answer"])
                
                result = self.evaluate_single_question(
                    question, gold_answer, thinking_effort, scale_factor, question_idx
                )
                
                self.save_result(result)
                config_results.append(result)
                self.results.append(result)

                if result["is_correct"]:
                    running_correct_count += 1
                running_total_tokens += result["total_tokens"]

                num_questions_so_far = len(config_results)
                current_accuracy = running_correct_count / num_questions_so_far
                avg_tokens_so_far = running_total_tokens / num_questions_so_far

                print(
                    f"  [Q {question_idx + 1}] "
                    f"Correct: {str(result['is_correct']):<5}. "
                    f"Score: {current_accuracy:.2f} ({running_correct_count}/{num_questions_so_far}). "
                    f"Tokens: {result['total_tokens']}. "
                    f"Avg Tokens: {avg_tokens_so_far:.1f}"
                )
            
            # Calculate configuration summary
            correct_count = sum(1 for r in config_results if r["is_correct"])
            accuracy = correct_count / len(config_results) if config_results else 0
            avg_thinking_tokens = sum(r["thinking_tokens"] for r in config_results) / len(config_results) if config_results else 0
            avg_total_tokens = sum(r["total_tokens"] for r in config_results) / len(config_results) if config_results else 0
            avg_generation_time = sum(r["generation_time"] for r in config_results) / len(config_results) if config_results else 0
            
            print(f"Configuration Results:")
            print(f"  Accuracy: {accuracy:.3f} ({correct_count}/{len(config_results)})")
            print(f"  Avg thinking tokens: {avg_thinking_tokens:.1f}")
            print(f"  Avg total tokens: {avg_total_tokens:.1f}")
            print(f"  Avg generation time: {avg_generation_time:.2f}s")
    
    def save_results_to_csv(self) -> str:
        """Saves summary of results to a new CSV file."""
        if not self.results:
            print("No results to save summary for.")
            return ""

        print(f"\nDetailed results have been saved to: {self.results_filename}")
        
        # Convert results to DataFrame for summary calculation
        df = pd.DataFrame(self.results)
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        
        # Group by configuration
        config_groups = df.groupby(["thinking_effort", "scale_factor"])
        
        summary_data = []
        for (thinking_effort, scale_factor), group in config_groups:
            accuracy = group["is_correct"].mean()
            avg_thinking_tokens = group["thinking_tokens"].mean()
            avg_total_tokens = group["total_tokens"].mean()
            avg_generation_time = group["generation_time"].mean()
            
            summary_data.append({
                "thinking_effort": thinking_effort,
                "scale_factor": scale_factor,
                "accuracy": accuracy,
                "avg_thinking_tokens": avg_thinking_tokens,
                "avg_total_tokens": avg_total_tokens,
                "avg_generation_time": avg_generation_time,
                "num_questions": len(group),
            })
        
        # Create summary DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_filename = self.results_filename.replace(".csv", "_summary.csv")
        summary_df.to_csv(summary_filename, index=False)
        
        print(f"Summary saved to: {summary_filename}")
        print(summary_df.to_string(index=False))
        
        return self.results_filename


def main():
    """Main evaluation function."""
    # Initialize evaluator
    evaluator = AIME2025Evaluator(
        model_name=EVALUATION_CONFIG["model_name"],
        dataset_name=EVALUATION_CONFIG["dataset_name"],
        dataset_config=EVALUATION_CONFIG["dataset_config"],
        dataset_split=EVALUATION_CONFIG["dataset_split"],
        max_new_tokens=EVALUATION_CONFIG["max_new_tokens"],
        temperature=EVALUATION_CONFIG["temperature"],
        top_p=EVALUATION_CONFIG["top_p"],
        top_k=EVALUATION_CONFIG["top_k"],
    )

    # Run evaluation
    evaluator.evaluate_configurations(
        thinking_efforts=EVALUATION_CONFIG["thinking_efforts"],
        scale_factors=EVALUATION_CONFIG["scale_factors"],
        max_questions=EVALUATION_CONFIG["max_questions"],
        include_baseline=EVALUATION_CONFIG.get("include_baseline", True),
    )

    # Save results
    evaluator.save_results_to_csv()


if __name__ == "__main__":
    main()
