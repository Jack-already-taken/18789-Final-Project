import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class VQADataset(Dataset):
    def __init__(self,
                 samples: list[dict],
                 vision_processor,
                 tokenizer: PreTrainedTokenizer):
        """
        samples: list of {"image_path": str, "question": str, "answer": str}
        """
        self.samples = samples
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # Load and preprocess image
        img = Image.open(entry["image_path"]).convert("RGB")
        pixel_values = self.vision_processor(
            images=img,
            return_tensors="pt"
        )["pixel_values"][0]  # shape: (3, H, W)

        # Tokenize question and answer
        q_ids = self.tokenizer(
            entry["question"],
            add_special_tokens=False
        ).input_ids
        a_ids = self.tokenizer(
            entry["answer"],
            add_special_tokens=False
        ).input_ids
        # Build concatenated inputs and labels mask
        input_ids = q_ids + a_ids
        labels    = [-100] * len(q_ids) + a_ids

        return (
            pixel_values,
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels,    dtype=torch.long)
        )


def prepare_vqa_data(hf_split,
                     images_root: str,
                     split: str = "train") -> list[dict]:
    """
    hf_split: HF dataset split (train or validation)
    images_root: base path containing train2014/ and val2014/
    split: "train" or "validation" (determines folder)

    Returns a list of dicts: {image_path, question, answer}
    """
    folder = "train2014" if split == "train" else "val2014"
    samples = []
    for example in hf_split:
        image_id = example["image_id"]
        # Construct filename
        if isinstance(image_id, int):
            filename = f"{image_id:012d}.jpg"
        else:
            filename = os.path.basename(image_id)
        image_path = os.path.join(images_root, folder, filename)

        # Extract the top-weighted label
        label_entry = example.get("label", {})
        ids_list     = label_entry.get("ids", [])
        weights_list = label_entry.get("weights", [])
        if ids_list and weights_list:
            max_idx = weights_list.index(max(weights_list))
            answer  = ids_list[max_idx]
        elif ids_list:
            answer = ids_list[0]
        else:
            answer = ""

        samples.append({
            "image_path": image_path,
            "question":   example["question"],
            "answer":     answer
        })
    return samples
