#!/usr/bin/env python3
"""
Evaluation script for ThinkingEffortProcessor on GSM8k benchmark.
Tests various thinking_effort and scale_factor configurations and saves detailed results to CSV.
"""

import re
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import itertools

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd

from thinking_effort_transformers import ThinkingEffortProcessor

EVALUATION_CONFIG = {
    "model_name": "Qwen/Qwen3-1.7B",
    "thinking_efforts": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "scale_factors": [1.5, 2.0, 2.5, 3.0, 4.0],
    "max_questions": 300,
    "max_new_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}


class GSM8kEvaluator:
    def __init__(self, model_name: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, device: str = "auto"):
        """
        Initialize the evaluator with a model and tokenizer.
        
        Args:
            model_name: HuggingFace model name
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
        
        # Load GSM8k dataset
        print("Loading GSM8k dataset...")
        self.dataset = load_dataset("gsm8k", "main", split="test")
        print(f"Loaded {len(self.dataset)} test examples")
        
        # Results storage
        self.results = []
        
    def find_end_thinking_token(self) -> int:
        """Find the token ID for the end-of-thinking marker."""
        # Common end-of-thinking tokens
        candidates = ["</think>", "</thinking>", "<|end_thinking|>"]
        
        for candidate in candidates:
            token_id = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_id) == 1:
                return token_id[0]
        
        # Fallback for QwQ model (known token ID)
        return 151668
    
    def extract_answer(self, text: str) -> Optional[float]:
        """Extract the final numerical answer from the model's response."""
        # Look for patterns like "The answer is X" or "#### X" (GSM8k format)
        patterns = [
            r"(?:the answer is|answer:|####)\s*([+-]?\d*\.?\d+)",
            r"(?:final answer|answer)\s*[:=]?\s*([+-]?\d*\.?\d+)",
            r"([+-]?\d*\.?\d+)\s*(?:is the answer|is the final answer)",
            r"\$([+-]?\d*\.?\d+)",  # Dollar amounts
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
        
        return None
    
    def extract_gold_answer(self, answer_text: str) -> float:
        """Extract the gold answer from GSM8k answer format."""
        # GSM8k answers are in format "Step-by-step solution\n#### final_answer"
        match = re.search(r"####\s*([+-]?\d*\.?\d+)", answer_text)
        if match:
            return float(match.group(1))
        
        # Fallback: look for any number at the end
        numbers = re.findall(r"([+-]?\d*\.?\d+)", answer_text)
        if numbers:
            return float(numbers[-1])
        
        raise ValueError(f"Could not extract answer from: {answer_text}")
    
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
            r"<\|thinking\|>(.*?)<\|end_thinking\|>",
        ]
        
        thinking_tokens = 0
        for pattern in thinking_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                thinking_tokens += len(self.tokenizer.encode(match))
        
        return thinking_tokens, total_tokens
    
    def generate_response(self, question: str, thinking_effort: float, scale_factor: float) -> Dict:
        """
        Generate a response to a question using the thinking effort processor.
        
        Returns:
            Dictionary with response details
        """
        # Create the thinking effort processor
        processor = ThinkingEffortProcessor(
            end_thinking_token_id=self.end_thinking_token_id,
            thinking_effort=thinking_effort,
            scale_factor=scale_factor
        )
        
        # Format the prompt for Qwen3 instruct model
        messages = [
            {"role": "user", "content": f"Please solve the following math problem. Your answer must end with the final numerical result in the format: ####<final answer>. \n\nProblem: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
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
                logits_processor=[processor],
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
    
    def evaluate_single_question(self, question: str, gold_answer: float, 
                               thinking_effort: float, scale_factor: float, 
                               question_idx: int) -> Dict:
        """Evaluate a single question with given parameters."""
        print(f"Evaluating question {question_idx + 1} with effort={thinking_effort}, scale={scale_factor}")
        
        try:
            result = self.generate_response(question, thinking_effort, scale_factor)
            
            # Check correctness
            is_correct = False
            if result["predicted_answer"] is not None:
                is_correct = abs(result["predicted_answer"] - gold_answer) < 1e-3
            
            return {
                "question_idx": question_idx,
                "question": question,
                "gold_answer": gold_answer,
                "thinking_effort": thinking_effort,
                "scale_factor": scale_factor,
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
                "thinking_effort": thinking_effort,
                "scale_factor": scale_factor,
                "predicted_answer": None,
                "is_correct": False,
                "thinking_tokens": 0,
                "total_tokens": 0,
                "generation_time": 0.0,
                "response": f"ERROR: {str(e)}",
            }
    
    def evaluate_configurations(self, 
                              thinking_efforts: List[float],
                              scale_factors: List[float],
                              max_questions: int = 100) -> None:
        """
        Evaluate all combinations of thinking_effort and scale_factor parameters.
        
        Args:
            thinking_efforts: List of thinking effort values to test
            scale_factors: List of scale factor values to test
            max_questions: Maximum number of questions to evaluate (for faster testing)
        """
        # Limit dataset size for testing
        test_dataset = self.dataset.select(range(min(max_questions, len(self.dataset))))
        
        # Generate all parameter combinations
        config_combinations = list(itertools.product(thinking_efforts, scale_factors))
        
        print(f"Testing {len(config_combinations)} configurations on {len(test_dataset)} questions")
        
        for config_idx, (thinking_effort, scale_factor) in enumerate(config_combinations):
            print(f"\n--- Configuration {config_idx + 1}/{len(config_combinations)} ---")
            print(f"Thinking effort: {thinking_effort}, Scale factor: {scale_factor}")
            
            config_results = []
            
            for question_idx, example in enumerate(test_dataset):
                question = example["question"]
                gold_answer = self.extract_gold_answer(example["answer"])
                
                result = self.evaluate_single_question(
                    question, gold_answer, thinking_effort, scale_factor, question_idx
                )
                
                config_results.append(result)
                self.results.append(result)
            
            # Calculate configuration summary
            correct_count = sum(1 for r in config_results if r["is_correct"])
            accuracy = correct_count / len(config_results)
            avg_thinking_tokens = sum(r["thinking_tokens"] for r in config_results) / len(config_results)
            avg_total_tokens = sum(r["total_tokens"] for r in config_results) / len(config_results)
            avg_generation_time = sum(r["generation_time"] for r in config_results) / len(config_results)
            
            print(f"Configuration Results:")
            print(f"  Accuracy: {accuracy:.3f} ({correct_count}/{len(config_results)})")
            print(f"  Avg thinking tokens: {avg_thinking_tokens:.1f}")
            print(f"  Avg total tokens: {avg_total_tokens:.1f}")
            print(f"  Avg generation time: {avg_generation_time:.2f}s")
    
    def save_results_to_csv(self, filename: Optional[str] = None) -> str:
        """Save all results to a CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gsm8k_thinking_effort_results_{timestamp}.csv"
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        print(f"\nResults saved to: {filename}")
        
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
        summary_filename = filename.replace(".csv", "_summary.csv")
        summary_df.to_csv(summary_filename, index=False)
        
        print(f"Summary saved to: {summary_filename}")
        print(summary_df.to_string(index=False))
        
        return filename


def main():
    """Main evaluation function."""
    # Initialize evaluator
    evaluator = GSM8kEvaluator(
        model_name=EVALUATION_CONFIG["model_name"],
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
    )

    # Save results
    evaluator.save_results_to_csv()


if __name__ == "__main__":
    main() 