from transformers import LogitsProcessor
import torch

class ThinkingEffortProcessor(LogitsProcessor):
    """
    Multiplies the logit for `end_thinking_token_id` by a factor
    determined by `thinking_effort`.
    
    Args:
        end_thinking_token_id: The token ID that marks the end of thinking.
        thinking_effort: A value that controls how much thinking to do.
            - thinking_effort=0 => scale=2 => more likely to pick the end token (less thinking)
            - thinking_effort=1 => scale=1 => no change
            - thinking_effort=2 => scale=0.5 => less likely to pick the end token (more thinking)
        scale_factor: Controls the intensity of the scaling effect (default=2).
            Higher values create a more dramatic difference between high and low
            thinking_effort settings.
    
    The formula used is: scale = scale_factor ** (1.0 - thinking_effort)
    """

    def __init__(self, end_thinking_token_id, thinking_effort=1.0, scale_factor=2):
        super().__init__()
        self.end_thinking_token_id = end_thinking_token_id
        self.thinking_effort = thinking_effort
        self.scale_factor = scale_factor

    def __call__(self, input_ids, scores):
        # 1) compute scaling factor from thinking_effort
        scale = self.scale_factor ** (1.0 - self.thinking_effort)
        # 2) multiply the logit for the end-thinking token
        scores[:, self.end_thinking_token_id] *= scale
        return scores