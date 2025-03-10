def thinking_effort_processor(thinking_effort, end_thinking_token_id, scale_factor=2):
    scale = scale_factor ** (1.0 - thinking_effort)
    token_generated = [False]

    def processor(input_ids, logits):
        if token_generated[0]:
            return logits

        # --- Convert "input_ids" to one integer for "last token" ---
        last_token_id = None

        # If it's just a Python list (common in some backends):
        if isinstance(input_ids, list):
            last_token_id = input_ids[-1]

        # If it's a NumPy array (common in llama_cpp):
        elif hasattr(input_ids, "shape"):
            # shape might be (seq_length,) or (batch, seq_length)
            if len(input_ids.shape) == 2:
                # get the last row's last element
                last_token_id = input_ids[-1, -1]
            else:
                # e.g. shape == (seq_length,)
                last_token_id = input_ids[-1]

            # convert that scalar array to a Python int
            last_token_id = int(last_token_id)

        else:
            # fallback if it's just a scalar or something else
            last_token_id = input_ids

        # --- Now compare ---
        if last_token_id == end_thinking_token_id:
            token_generated[0] = True
            return logits

        # If we haven't yet seen the end_thinking_token, scale its logit
        logits[end_thinking_token_id] *= scale
        return logits

    return processor
