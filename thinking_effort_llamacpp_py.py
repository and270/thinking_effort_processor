def thinking_effort_processor(thinking_effort, end_thinking_token_id, scale_factor=2):
    """
    Returns a function that scales the logit of the end_thinking_token_id
    based on the specified thinking_effort.
    
    Args:
        thinking_effort: A value between 0 and 1 that controls how much thinking to do.
            Higher values (closer to 1) encourage more thinking by reducing the probability
            of the end_thinking_token.
            Lower values (closer to 0) encourage less thinking by increasing the probability
            of the end_thinking_token.
        end_thinking_token_id: The token ID that marks the end of thinking.
        scale_factor: Controls the intensity of the scaling effect (default=2).
            Higher values create a more dramatic difference between high and low
            thinking_effort settings.
    
    Returns:
        A logit processor function that modifies the probability of the end_thinking_token.
    """
    scale = scale_factor ** (1.0 - thinking_effort)

    def processor(input_ids, logits):
        logits[end_thinking_token_id] *= scale
        return logits

    return processor
