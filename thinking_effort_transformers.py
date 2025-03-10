from transformers import LogitsProcessor
import torch

class ThinkingEffortProcessor(LogitsProcessor):
    """
    Multiplies the logit for `end_thinking_token_id` by a factor
    determined by `thinking_effort` in [0,2].
    
    - thinking_effort=0 => scale=2 => more likely to pick the end token (less thinking)
    - thinking_effort=1 => scale=1 => no change
    - thinking_effort=2 => scale=0.5 => less likely to pick the end token (more thinking)
    """

    def __init__(self, end_thinking_token_id, thinking_effort=1.0):
        super().__init__()
        self.end_thinking_token_id = end_thinking_token_id
        self.thinking_effort = thinking_effort

    def __call__(self, input_ids, scores):
        # 1) compute scaling factor from thinking_effort
        scale = 2 ** (1.0 - self.thinking_effort)
        # 2) multiply the logit for the end-thinking token
        scores[:, self.end_thinking_token_id] *= scale
        return scores