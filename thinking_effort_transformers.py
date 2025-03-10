from transformers import LogitsProcessor
import torch

class ThinkingEffortProcessor(LogitsProcessor):
    """
    A custom LogitsProcessor for Hugging Face Transformers that scales the logit for an
    "end-of-thinking" token based on a `thinking_effort` parameter—until that token is
    actually generated for each sequence, at which point it stops scaling for that sequence.

    Args:
        end_thinking_token_id (int):
            The special token ID representing the end-of-thinking marker (e.g. </think>).
        thinking_effort (float, optional):
            Controls how heavily to scale the end_thinking_token_id. Interpreted via:
                scale = scale_factor ** (1.0 - thinking_effort)
            - If thinking_effort=0, scale=scale_factor^1 => strongly boosts the end token 
              (reducing thinking).
            - If thinking_effort=1, scale=scale_factor^0 => no scaling on the end token
              (normal chance, i.e. more thinking).
            - If thinking_effort>1, scale<1 => end token is suppressed (extensive thinking).
            Default is 1.0.
        scale_factor (float, optional):
            The base used in the exponent that determines how strongly to scale the end token.
            Default is 2.

    Behavior:
        - For each sequence in a batch, if the end_thinking_token_id has already appeared
          in previous steps of generation, no further scaling is applied for that sequence.
        - Otherwise, the logit for the end_thinking_token_id is multiplied by `scale`.


    Explanation:
        - The code runs at each generation step. For each sequence (row) in the batch:
            1. If that sequence has already generated `end_thinking_token_id`, do nothing.
            2. Otherwise, scale that token's logit by `scale = scale_factor ** (1.0 - thinking_effort)`.
        - This makes the end token more or less likely to appear, depending on `thinking_effort`.
    """

    def __init__(self, end_thinking_token_id, thinking_effort=1.0, scale_factor=2):
        super().__init__()
        self.end_thinking_token_id = end_thinking_token_id
        self.thinking_effort = thinking_effort
        self.scale_factor = scale_factor
        # Track which sequences (by index) have already produced the end_thinking_token_id
        self.finished_sequences = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Invoked at each generation step by the Transformers library.

        Args:
            input_ids (torch.LongTensor):
                The tokens generated so far, shape (batch_size, seq_length).
            scores (torch.FloatTensor):
                The current logits for the next token, shape (batch_size, vocab_size).

        Returns:
            torch.FloatTensor:
                The modified logits (same shape) with the end_thinking_token_id scaled
                for sequences that have not yet generated it.
        """
        # Compute the scale factor from the current thinking_effort
        scale = self.scale_factor ** (1.0 - self.thinking_effort)

        batch_size = input_ids.size(0)
        # For each sequence in the batch, check if we've generated the end token before
        for i in range(batch_size):
            if i in self.finished_sequences:
                # Do not scale if we've already seen the end_thinking_token for this sequence
                continue
            
            # Check if this sequence contains the end_thinking_token_id already
            if (input_ids[i] == self.end_thinking_token_id).any():
                # Mark that we've generated the end token for this sequence
                self.finished_sequences.add(i)
                # Don't scale the logit anymore
                continue

            # If we haven't encountered it yet, scale the logit for the end_thinking_token
            scores[i, self.end_thinking_token_id] *= scale

        return scores
