# train_coco.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models import (
    vision_model,
    visual_projector,
    lm_model,
    forward_multimodal,
    device,
)
from coco_dataset import CocoCaptionDataset, processor, tokenizer, collate_fn, examples

# ── hyper‑parameters ──────────────────────────────────────────────────────────
BATCH_SIZE, NUM_EPOCHS, SAVE_EVERY = 32, 10, 1
LR, GRAD_CLIP, NUM_WORKERS         = 1e-4, 1.0, 4
CKPT_DIR                          = "coco_checkpoints"
# ───────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1) build DataLoader
    dataset    = CocoCaptionDataset(examples, processor, tokenizer, max_length=64)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 2) optimizer & scaler (only projector is trainable)
    optimizer = AdamW(visual_projector.parameters(), lr=LR)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    vision_model.eval()
    lm_model.eval()
    visual_projector.train()

    for epoch in range(1, NUM_EPOCHS+1):
        total_loss = 0.0
        for pix, caps in tqdm(train_loader, desc=f"Epoch {epoch}"):
            pix, caps = pix.to(device), caps.to(device)
            input_ids = caps
            labels    = caps.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # prepend mask for the visual prefix
            prefix = torch.full((labels.size(0),1), -100, device=device, dtype=torch.long)
            labels = torch.cat([prefix, labels], dim=1)

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits      = forward_multimodal(pix, input_ids)        # (B, T+1, V)
                logits_s    = logits[:, :-1, :].contiguous()
                labels_s    = labels[:, 1:].contiguous()
                loss        = F.cross_entropy(
                                  logits_s.view(-1, logits_s.size(-1)),
                                  labels_s.view(-1),
                                  ignore_index=-100
                              )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(visual_projector.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}  avg loss: {avg_loss:.4f}")

        # save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            ckpt_path = os.path.join(CKPT_DIR, f"visual_projector_ep{epoch:02d}.pt")
            torch.save(visual_projector.state_dict(), ckpt_path)
            print(f"✓ saved {ckpt_path}")

    # final save
    final_path = os.path.join(CKPT_DIR, "visual_projector_final.pt")
    torch.save(visual_projector.state_dict(), final_path)
    print(f"✓ final projector saved to {final_path}")

if __name__ == "__main__":
    main()
