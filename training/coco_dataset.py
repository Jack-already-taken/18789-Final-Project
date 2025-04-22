# coco_dataloader.py
import os
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoProcessor, AutoTokenizer

# ─── 1) Paths ─────────────────────────────────────────────────────────────────
data_root = Path("/scratch/coco2017")
img_dir   = data_root / "train2017"
ann_file  = data_root / "annotations" / "captions_train2017.json"

# ─── 2) Build (image_path, caption) examples ───────────────────────────────────
with open(ann_file, "r") as f:
    ann_json = json.load(f)

# map image_id → file_name
images_map = { img["id"]: img["file_name"] for img in ann_json["images"] }

examples: List[Dict] = []
for ann in ann_json["annotations"]:
    img_id = ann["image_id"]
    fname  = images_map[img_id]
    path   = img_dir / fname
    examples.append({
        "image_path": str(path),
        "caption":     ann["caption"]
    })

# ─── 3) Dataset ────────────────────────────────────────────────────────────────
class CocoCaptionDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict],
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        max_length: int = 64
    ):
        self.examples   = examples
        self.processor  = processor
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        # load & preprocess image
        img = Image.open(ex["image_path"]).convert("RGB")
        pix = self.processor(images=img, return_tensors="pt")\
                    .pixel_values.squeeze(0)         

        # tokenize caption (no padding here)
        caps = self.tokenizer(
            ex["caption"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False
        ).input_ids.squeeze(0)                         # (L,)

        return pix, caps

# ─── 4) Instantiate processor & tokenizer ────────────────────────────────────
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    use_fast=False
)

# ─── 5) Collate function ─────────────────────────────────────────────────────
def collate_fn(batch):
    pixs, caps = zip(*batch)
    pixs = torch.stack(pixs, dim=0)  # (B, 3, H, W)

    caps = pad_sequence(
        caps,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )                                 # (B, L_max)

    return pixs, caps

# ─── 6) DataLoader ───────────────────────────────────────────────────────────
dataset   = CocoCaptionDataset(examples, processor, tokenizer)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

# ─── 7) Sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pix_batch, cap_batch = next(iter(dataloader))
    
    # decode the first 5 captions
    decoded = [tokenizer.decode(ids, skip_special_tokens=True) 
               for ids in cap_batch[:5]]
    
    # print out the token IDs and decoded strings
    for i, (ids, text) in enumerate(zip(cap_batch[:5], decoded)):
        print(f"Example {i}")
        print("  Token IDs:", ids.tolist())
        print("  Decoded :", text)
        print()
    
    # plot the first 5 images
    fig, axes = plt.subplots(1, 5, figsize=(15,3))
    for i, ax in enumerate(axes):
        img = pix_batch[i].permute(1,2,0).cpu().numpy()  # HWC
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"#{i}")
    plt.tight_layout()
    plt.show()
