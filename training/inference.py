# inference.py
import torch
from PIL import Image

from models import (
    vision_processor,
    forward_multimodal,
    lm_model,
    vision_model,
    visual_projector,
    device,
    tokenizer
)

# 1) Load your trained projector
state = torch.load("visual_projector.pt", map_location=device)
visual_projector.load_state_dict(state)
print("   state keys:", list(state.keys()))
# 2) Move all modules to device & set eval mode
vision_model.to(device).eval()
visual_projector.to(device).eval()
lm_model.to(device).eval()

@torch.no_grad()
def generate_answer(image_path: str, question: str, max_new_tokens=50):
    # Preprocess image
    img = Image.open(image_path).convert("RGB")
    pix = vision_processor(images=img, return_tensors="pt")["pixel_values"].to(device)

    # Tokenize question
    q = tokenizer(question, return_tensors="pt").input_ids.to(device)

    # 1) run multimodal prefix to get past_key_values
    out = forward_multimodal(pix, q)
    past = out.past_key_values

    # 2) greedy decode
    next_token = torch.tensor([[ out.logits.argmax(-1)[0, -1] ]], device=device)
    generated = []
    for _ in range(max_new_tokens):
        out = lm_model(input_ids=next_token, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token.item())
        past = out.past_key_values
    print("Generated Token IDs:", generated)
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    ans = generate_answer("test.jpg", "What is the dog doing?")
    print("Answer:", ans)
