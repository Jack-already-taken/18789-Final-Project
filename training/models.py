# models.py
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoImageProcessor, AutoTokenizer

# 1. Device setup
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# 2. CLIP vision backbone + processor
# VISION_MODEL_NAME = "openai/clip-vit-large-patch14"
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"
vision_model = CLIPVisionModel.from_pretrained(VISION_MODEL_NAME).to(device)
vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)

for p in vision_model.parameters():
    p.requires_grad = False

# 3. Language model + tokenizer
# LM_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LM_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

lm_model = AutoModelForCausalLM.from_pretrained(LM_NAME, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(LM_NAME, use_fast=False)

for p in lm_model.parameters():
    p.requires_grad = False

# 4. Projection head
clip_dim = vision_model.config.hidden_size
qwen_dim = lm_model.config.hidden_size
def SimpleResBlock(dim: int):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Linear(dim, dim),
        # residual add will be applied manually
    )
class ProjectionWithResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj     = nn.Linear(in_dim, out_dim)
        self.resblock = SimpleResBlock(out_dim)
    def forward(self, x):
        x0 = self.proj(x)
        x1 = self.resblock(x0)
        return x0 + x1

visual_projector = ProjectionWithResBlock(clip_dim, qwen_dim).to(device)

def forward_multimodal(image_tensor: torch.Tensor, question_ids: torch.LongTensor):
    """
    image_tensor: (B, 3, H, W)
    question_ids: (B, T)
    """
    # 1. CLIP features
    with torch.no_grad():
        vis_out = vision_model(pixel_values=image_tensor.to(device))
    img_feat = vis_out.last_hidden_state[:, 0, :]  # (B, clip_dim)

    # 2. Project
    img_emb = visual_projector(img_feat)           # (B, qwen_dim)

    # 3. Token embeddings
    text_embeds = lm_model.get_input_embeddings()(question_ids.to(device))

    # 4. Prepend img_emb as a “prefix token”
    combined = torch.cat([img_emb.unsqueeze(1), text_embeds], dim=1)
    seq_len = combined.size(1)
    attention_mask = combined.new_ones((combined.size(0), seq_len))
    position_ids    = torch.arange(seq_len, device=device).unsqueeze(0)
    # 5. Forward through LM
    out = lm_model(
        inputs_embeds=combined,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
    )
    return out.logits  
