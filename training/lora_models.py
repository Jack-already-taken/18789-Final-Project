# models.py
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoImageProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Device
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# 2. Load & freeze CLIP vision backbone
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"
vision_model = CLIPVisionModel.from_pretrained(VISION_MODEL_NAME).to(device)
vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
for p in vision_model.parameters():
    p.requires_grad = False

# 3. Load & freeze LLM
LM_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lm_model = AutoModelForCausalLM.from_pretrained(LM_NAME, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(LM_NAME, use_fast=False)
for p in lm_model.parameters():
    p.requires_grad = False

# 4. Define your projector & freeze base
clip_dim = vision_model.config.hidden_size
qwen_dim = lm_model.config.hidden_size

class SimpleResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.net(self.norm(x))

class ProjectionWithResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj     = nn.Linear(in_dim, out_dim)
        self.resblock = SimpleResBlock(out_dim)
    def forward(self, x):
        return self.resblock(self.proj(x))

visual_projector = ProjectionWithResBlock(clip_dim, qwen_dim).to(device)


# 5. Create LoRA configs
lora_config_llm = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,               # rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
lora_config_vision = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    target_modules=["proj"],  # name of the Linear inside ProjectionWithResBlock
)

# 6. Wrap with PEFT → only these adapters get trained
lm_model           = get_peft_model(lm_model, lora_config_llm)

# 7. Forward fn stays the same
def forward_multimodal(image_tensor: torch.Tensor, question_ids: torch.LongTensor):
    # CLIP features (frozen)
    with torch.no_grad():
        vis_out = vision_model(pixel_values=image_tensor)
    img_feat = vis_out.last_hidden_state[:, 0, :]         # (B, clip_dim)

    # vision adapter (LoRA‑augmented)
    img_emb  = visual_projector(img_feat)                 # (B, qwen_dim)
    text_emb = lm_model.get_input_embeddings()(question_ids.to(device))
    combined = torch.cat([img_emb.unsqueeze(1), text_emb], dim=1)
    mask     = combined.new_ones((combined.size(0), combined.size(1)))
    pos_ids  = torch.arange(combined.size(1), device=device).unsqueeze(0)

    out = lm_model(
        inputs_embeds   = combined,
        attention_mask  = mask,
        position_ids    = pos_ids,
        return_dict=True,
    )
    return out.logits
