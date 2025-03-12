def thinking_effort_processor(thinking_effort, end_thinking_token_id, scale_factor=2):
    """
    Creates a callable logit-processor that modifies the probability of an 'end thinking' token
    based on the specified thinking effort. Typically used with llama-cpp or similar backends
    that support a custom logits_processor.

    Args:
        thinking_effort (float):
            A value between 0 and 1 that controls how much thinking to do.
            - Higher values (closer to 1) encourage more thinking by reducing the probability
              scaling of the end_thinking_token_id (i.e., minimal scaling).
            - Lower values (closer to 0) encourage less thinking by more strongly scaling up
              the probability of the end_thinking_token_id (i.e., large scaling).
        end_thinking_token_id (int):
            The token ID that marks the end of the "thinking" phase (e.g. </think>). On QwQ model, for example, this is 151668.
        scale_factor (float, optional):
            Controls the intensity of the scaling effect (default=2).
            - At thinking_effort=0.0, the end_thinking_token_id logit is multiplied by
              scale_factor^(1 - 0.0) = scale_factor (forcing the end token more likely).
            - At thinking_effort=1.0, the end_thinking_token_id logit is multiplied by
              scale_factor^(1 - 1.0) = 1 (no extra forcing).

    Returns:
        function:
            A logit processor function with signature (input_ids, logits) -> logits.

    Implementation Details:
        - The returned processor examines the most recent token in `input_ids`. If it matches
          `end_thinking_token_id`, we record that the end-of-thinking token has been generated
          and cease further scaling.
        - Because llama-cpp may pass `input_ids` as either a Python list or a NumPy array (possibly
          with shape (batch, seq_length)), the processor carefully extracts the last token as a
          Python int. This avoids errors when comparing it to `end_thinking_token_id`.
        - Once the end token is generated, the processor stops modifying logits altogether.
    """
    # Compute how strongly to scale the end_thinking_token_id
    scale = scale_factor ** (1.0 - thinking_effort)

    # We store the "has generated end token" state in a list to make it mutable in this closure
    token_generated = [False]

    def processor(input_ids, logits):
        """
        This inner function is the actual logit processor used at generation time.
        
        Args:
            input_ids: Could be a Python list of token IDs or a NumPy array of shape
                       (seq_length,) or (batch_size, seq_length).
            logits:    A 1D array or similar structure with the current token logits.

        Returns:
            Modified logits with the end_thinking_token scaled unless we've already
            seen that token.
        """
        # If we've already generated the end token, do nothing further
        if token_generated[0]:
            return logits

        # Convert "input_ids" to a single integer for the last token
        if isinstance(input_ids, list):
            last_token_id = input_ids[-1]
        elif hasattr(input_ids, "shape"):
            # shape could be (seq_length,) or (batch, seq_length)
            if len(input_ids.shape) == 2:
                # Last row's last element
                last_token_id = input_ids[-1, -1]
            else:
                # e.g. shape == (seq_length,)
                last_token_id = input_ids[-1]
            # Convert from scalar array to Python int
            last_token_id = int(last_token_id)
        else:
            # fallback if it's just a scalar or another structure
            last_token_id = input_ids

        # If we've just generated the end_thinking_token, record that fact and do no more scaling
        if last_token_id == end_thinking_token_id:
            token_generated[0] = True
            return logits

        # Otherwise, multiply its logit by the scale factor
        logits[end_thinking_token_id] *= scale
        return logits

    return processor
