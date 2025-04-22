# train.py
import os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from lora_models import (
    vision_model, visual_projector, lm_model,
    vision_processor, tokenizer,
    forward_multimodal, device,
)
from dataset import VQADataset, prepare_vqa_data
from datasets import load_dataset

# ── hyper‑params ─────────────────────────────────────────────────────────────
BATCH_SIZE, NUM_EPOCHS, SAVE_EVERY = 64, 10, 1
LR, GRAD_CLIP, NUM_WORKERS   = 1e-4, 1.0, 8
# ───────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    pixs, in_ids, labs = zip(*batch)
    pixs = torch.stack(pixs, dim=0)
    in_ids_padded = pad_sequence(in_ids, batch_first=True,
                                 padding_value=tokenizer.pad_token_id)
    labs_padded   = pad_sequence(labs, batch_first=True, padding_value=-100)
    return pixs, in_ids_padded, labs_padded

def main():
    # prepare data
    hf_train = load_dataset("Graphcore/vqa", split="train", trust_remote_code=True)
    samples  = prepare_vqa_data(hf_train, images_root="images", split="train")
    vqa_ds   = VQADataset(samples, vision_processor, tokenizer)
    vqa_dl   = DataLoader(
        vqa_ds, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, collate_fn=collate_fn,
    )

    # only LoRA params have requires_grad=True
    optimizer = AdamW(
        list(lm_model.parameters()) + list(visual_projector.parameters()), lr=LR
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    vision_model.eval()
    lm_model.train()
    visual_projector.train()

    print(f"Training {NUM_EPOCHS} epochs – batch {BATCH_SIZE} on {device}")
    ckpt_dir = "lora_checkpoints"; os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS+1):
        running_loss = 0.0
        for step, (pix, input_ids, labels) in enumerate(tqdm(vqa_dl, desc=f"Epoch {epoch}")):
            pix, input_ids, labels = (
                pix.to(device, non_blocking=True),
                input_ids.to(device),
                labels.to(device),
            )
            prefix_mask = torch.full((labels.size(0),1), -100,
                                     dtype=labels.dtype, device=device)
            labels = torch.cat([prefix_mask, labels], dim=1)

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits      = forward_multimodal(pix, input_ids)
                logits_shift= logits[:, :-1, :].contiguous()
                labels_shift= labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    logits_shift.view(-1, logits_shift.size(-1)),
                    labels_shift.view(-1),
                    ignore_index=-100,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(visual_projector.parameters()) + list(lm_model.parameters()),
                GRAD_CLIP
            )
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
        avg = running_loss / len(vqa_dl)
        print(f"Epoch {epoch} – avg loss {avg:.4f}")

        # 1) standard projector checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            path = os.path.join(ckpt_dir, f"visual_projector_ep{epoch:02d}.pt")
            torch.save(visual_projector.state_dict(), path)
            print(f"✓ saved {path}")

            # 2) LoRA‑only adapters
            lora_dir = os.path.join(ckpt_dir, f"checkpoint-lora_ep{epoch:02d}")
            os.makedirs(os.path.join(lora_dir, "llm_adapter"), exist_ok=True)

            lm_model.save_pretrained(os.path.join(lora_dir, "llm_adapter"))
            print(f"✓ saved LoRA adapters at {lora_dir}/")


    # final projector
    torch.save(
        visual_projector.state_dict(),
        os.path.join(ckpt_dir, "visual_projector_final.pt")
    )
    print("✓ final projector saved")

if __name__ == "__main__":
    main()
