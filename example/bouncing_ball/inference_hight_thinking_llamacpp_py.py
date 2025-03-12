import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llama_cpp import Llama
from thinking_effort_llamacpp_py import thinking_effort_processor

model_path = "C:/Users/andre/.cache/lm-studio/models/lmstudio-community/QwQ-32B-GGUF/QwQ-32B-Q6_K.gguf"

llm = Llama(model_path=model_path, n_ctx=131072)

thinking_effort = 2.5 

# Get the token ID for the '</think>' token
end_thinking_token_id = 151668   #</think> token id for QwQ model

processor = thinking_effort_processor(thinking_effort, end_thinking_token_id)
logits_processor = [processor]

#IMPORTANT: chat template for Qwen model. If other model, you must change the prompt format.
prompt = """<|im_start|>user
Write a Python program that shows 20 balls bouncing inside a spinning heptagon:
- All balls have the same radius.
- All balls have a number on it from 1 to 20.
- All balls drop from the heptagon center when starting, but they are not in the exact same spot, they are randomly distributed.
- Colors are: #f8b862, #f6ad49, #f39800, #f08300, #ec6d51, #ee7948, #ed6d3d, #ec6800, #ec6800, #ee7800, #eb6238, #ea5506, #ea5506, #eb6101, #e49e61, #e45e32, #e17b34, #dd7a56, #db8449, #d66a35
- The balls should be affected by gravity and friction, and they must bounce off the rotating walls realistically. There should also be collisions between balls.
- The material of all the balls determines that their impact bounce height will not exceed the radius of the heptagon, but higher than ball radius.
- All balls rotate with friction, the numbers on the ball can be used to indicate the spin of the ball.
- The heptagon is spinning around its center, and the speed of spinning is 360 degrees per 5 seconds.
- The heptagon size should be large enough to contain all the balls.
- Do not use the pygame library; implement collision detection algorithms and collision response etc. by yourself. The following Python libraries are allowed: tkinter, math, numpy, dataclasses, typing, sys.
- All codes should be put in a single Python file.
- The balls should remain inside the heptagon.
- If the balls hit the heptagon walls, they should bounce off realistically.
<|im_end|>
<|im_start|>assistant
<think>
"""

# Open a file to save the response
output_file = "llm_response.txt"
with open(output_file, "w", encoding="utf-8") as f:

    print("Streaming output with thinking effort (response will also be saved in llm_response.txt):")
    for chunk in llm.create_completion(
        prompt,
        max_tokens=64000,
        temperature=0.6,
        logits_processor=logits_processor,
        stream=True  # Enable streaming
    ):
        chunk_text = chunk['choices'][0]['text']

        print(chunk_text, end='', flush=True)

        f.write(chunk_text)
        f.flush()  # Make sure content is written immediately


print()
print(f"Full response has been saved to {output_file}")