def thinking_effort_processor(thinking_effort, end_thinking_token_id):
    """
    Returns a function that scales the logit of the end_thinking_token_id
    based on the specified thinking_effort.
    """
    scale = 2 ** (1.0 - thinking_effort)

    def processor(input_ids, logits):
        logits[end_thinking_token_id] *= scale
        return logits

    return processor
