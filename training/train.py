# train.py  ── updated to use next‑token prediction and new torch.amp API
import os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models import (
    vision_model, visual_projector, lm_model, vision_processor, tokenizer,
    forward_multimodal, device,
)
from dataset import VQADataset, prepare_vqa_data
from datasets import load_dataset
from torch import amp                                # <-- new API

# ── hyper‑params ─────────────────────────────────────────────────────────────
BATCH_SIZE, NUM_EPOCHS, SAVE_EVERY = 64, 10, 1
LR, GRAD_CLIP, NUM_WORKERS = 1e-4, 1.0, 8
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    pixs, in_ids, labs = zip(*batch)
    pixs = torch.stack(pixs, dim=0)
    in_ids_padded = pad_sequence(in_ids, batch_first=True,
                                 padding_value=tokenizer.pad_token_id)
    labs_padded   = pad_sequence(labs, batch_first=True, padding_value=-100)
    return pixs, in_ids_padded, labs_padded

# ── training loop ────────────────────────────────────────────────────────────
def main():
    hf_train = load_dataset("Graphcore/vqa", split="train", trust_remote_code=True)
    samples  = prepare_vqa_data(hf_train, images_root="images", split="train")

    vqa_ds = VQADataset(samples, vision_processor, tokenizer)
    vqa_dl = DataLoader(vqa_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True, collate_fn=collate_fn)
    
    optimizer = AdamW(visual_projector.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    vision_model.eval(); lm_model.eval(); visual_projector.train()

    # if args.resume:
    #     ckpt = torch.load(args.resume, map_location=device)
    #     visual_projector.load_state_dict(ckpt['model_state_dict'])
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #     scaler.load_state_dict(ckpt['scaler_state_dict'])
    #     start_epoch = ckpt['epoch'] + 1
    #     print(f"Resuming from {args.resume}, starting at epoch {start_epoch}")

    print(f"Training {NUM_EPOCHS} epochs – batch {BATCH_SIZE} on {device}")

    ckpt_dir = "checkpoints"; os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0
        for step, (pix, input_ids, labels) in enumerate(
                tqdm(vqa_dl, desc=f"Epoch {epoch}")):

            pix, input_ids, labels = (pix.to(device, non_blocking=True),
                                      input_ids.to(device),
                                      labels.to(device))

            prefix_mask = torch.full((labels.size(0), 1), -100,
                                     dtype=labels.dtype, device=device)
            labels = torch.cat([prefix_mask, labels], dim=1)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits     = forward_multimodal(pix, input_ids)
                                     # (B, L+1, V)

                # shift: predict t+1 token at position t
                logits_shift = logits[:, :-1, :].contiguous()
                labels_shift = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    logits_shift.view(-1, logits_shift.size(-1)),
                    labels_shift.view(-1),
                    ignore_index=-100,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(visual_projector.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            if step and step % 1000 == 0:
                print(f"  step {step:>6}  avg loss: {running_loss / step:.4f}")

        print(f"Epoch {epoch}/{NUM_EPOCHS} – avg loss {running_loss / len(vqa_dl):.4f}")

        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            path = os.path.join(ckpt_dir, f"visual_projector_ep{epoch:02d}.pt")
            torch.save(visual_projector.state_dict(), path)
            print(f"✓ saved {path}")

    torch.save(visual_projector.state_dict(),
               os.path.join(ckpt_dir, "visual_projector_final.pt"))
    print("✓ final projector saved")

if __name__ == "__main__":
    main()
