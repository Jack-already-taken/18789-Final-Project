# inference.py
import torch
from PIL import Image

from models import (
    vision_model,
    visual_projector,
    lm_model,
    vision_processor,
    tokenizer,
    device,
)
import os

# ──────────────────────────────────────────────────────────────────────────────
# Load the fine‑tuned projector weights
# ──────────────────────────────────────────────────────────────────────────────
CKPT = "/home/zhuominc/johnz/vlm/checkpoints/visual_projector_ep05.pt"
state = torch.load(CKPT, map_location=device)
visual_projector.load_state_dict(state, strict=True)
print(f"✓ loaded projector checkpoint – {len(state)} tensors")

vision_model.to(device).eval()
visual_projector.to(device).eval()
lm_model.to(device).eval()

# helpers
EOS = tokenizer.eos_token_id


@torch.no_grad()
def _encode_with_prefix(img_tensor: torch.Tensor, question_ids: torch.LongTensor):
    """
    Build the visual‑text prefix and get (logits, past_key_values).
    Returns: last_token (LongTensor shape 1×1), past (list of tuples)
    """
    # 1. CLIP CLS embedding
    vis_out  = vision_model(pixel_values=img_tensor)
    img_feat = vis_out.last_hidden_state[:, 0, :]          # (1, clip_dim)

    # 2. Project to LLM hidden size
    img_emb  = visual_projector(img_feat)                  # (1, qwen_dim)

    # 3. Question token embeddings
    txt_emb  = lm_model.get_input_embeddings()(question_ids)

    # 4. Concat  →  (1, 1+T, dim)
    combined = torch.cat([img_emb.unsqueeze(1), txt_emb], dim=1)

    B, S, _  = combined.shape
    attn_mask = combined.new_ones((B, S))
    pos_ids   = torch.arange(S, device=device).unsqueeze(0)

    # 5. Forward through the LM to get cache
    out = lm_model(
        inputs_embeds=combined,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        use_cache=True,
        return_dict=True,
    )

    last_token = out.logits.argmax(-1)[:, -1:]             # shape (1,1)
    return last_token, out.past_key_values


@torch.no_grad()
def generate_answer(image_path: str, question: str, max_new_tokens: int = 50):
    # preprocess image
    img = Image.open(image_path).convert("RGB")
    pix = vision_processor(images=img, return_tensors="pt").pixel_values.to(device)

    # question ids
    q_ids = tokenizer(question, return_tensors="pt" ).input_ids.to(device)

    # initial forward pass
    next_tok, past = _encode_with_prefix(pix, q_ids)
    generated = []

    for i in range(max_new_tokens):
        out = lm_model(input_ids=next_tok, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        next_tok = logits.argmax(-1, keepdim=True)

        # if next_tok.item() == EOS:
        #     break
        # if i>0 and next_tok.item() == EOS:
        #     break
        generated.append(next_tok.item())
        past = out.past_key_values

    print("Generated token ids:", generated)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# quick CLI test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ans = generate_answer("test3.jpg", "What is this a picture of?")
    print("Answer:", ans)
    ans = generate_answer("test3.jpg", "How many birds are in this image?")
    print("Answer:", ans)
    ans = generate_answer("test3.jpg", "What kind of bird is this?")
    print("Answer:", ans)
