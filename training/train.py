import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models import (
    vision_model,
    visual_projector,
    lm_model,
    vision_processor,
    tokenizer,
    forward_multimodal,
    device,
)
from dataset import VQADataset, prepare_vqa_data
from datasets import load_dataset

# ──────────────────────────────────────────────────────────────────────────────
# Hyper‑parameters & constants
# ──────────────────────────────────────────────────────────────────────────────

BATCH_SIZE   = 24
NUM_EPOCHS   = 2000          # set however many you like
SAVE_EVERY   = 1000          # ← save a checkpoint every N epochs
LR           = 1e-4
GRAD_CLIP    = 1.0
NUM_WORKERS  = 8

# ──────────────────────────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Batch elements: (pixel_values, input_ids, labels)."""
    pixs, in_ids, labs = zip(*batch)

    # Stack images  ➜  (B, 3, H, W)
    pixs = torch.stack(pixs, dim=0)

    # Pad question & label sequences
    in_ids_padded = pad_sequence(
        in_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labs_padded = pad_sequence(labs, batch_first=True, padding_value=-100)
    return pixs, in_ids_padded, labs_padded


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load raw VQA annotations (≈1 GB)
    hf_train = load_dataset("Graphcore/vqa", split="train", trust_remote_code=True)

    # 2) Build list of samples (image‑path, question_ids, label_ids)
    samples = prepare_vqa_data(hf_train, images_root="images", split="train")

    # 3) Create torch‑style dataset & dataloader
    vqa_ds = VQADataset(samples, vision_processor, tokenizer)
    vqa_dl = DataLoader(
        vqa_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    # 4) Only projector is trainable
    optimizer = AdamW(visual_projector.parameters(), lr=LR)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Freeze encoders; projector in train‑mode
    vision_model.eval()
    lm_model.eval()
    visual_projector.train()

    print(f"Training for {NUM_EPOCHS} epochs ‑ batch {BATCH_SIZE} on {device}…")

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0
        for step, (pix, input_ids, labels) in enumerate(tqdm(vqa_dl, desc=f"Epoch {epoch}")):
            pix, input_ids, labels = (
                pix.to(device, non_blocking=True),
                input_ids.to(device),
                labels.to(device),
            )

            # Prepend −100 label for visual prefix token
            prefix_mask = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=device)
            labels = torch.cat([prefix_mask, labels], dim=1)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = forward_multimodal(pix, input_ids)  # (B, L+1, V)
                loss   = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(visual_projector.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            if step and step % 1000 == 0:
                print(f"  step {step:>6}  avg loss: {running_loss / step:.4f}")

        epoch_loss = running_loss / len(vqa_dl)
        print(f"Epoch {epoch}/{NUM_EPOCHS} complete – avg loss {epoch_loss:.4f}")

        # ── checkpoint ────────────────────────────────────────────────────────
        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"visual_projector_ep{epoch:06d}.pt")
            torch.save(visual_projector.state_dict(), ckpt_path)
            print(f"✓ Saved checkpoint → {ckpt_path}")

    # Final save (redundant if NUM_EPOCHS multiple of SAVE_EVERY)
    final_path = os.path.join(ckpt_dir, "visual_projector_final.pt")
    torch.save(visual_projector.state_dict(), final_path)
    print(f"✓ Saved final projector → {final_path}")


if __name__ == "__main__":
    main()
