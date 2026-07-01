import os
from pathlib import Path

import torch
import torch.optim as optim
import torch.amp
import torchvision.utils as vutils
from tqdm import tqdm

from dataset import build_train_loader
from model   import RetinexVideoEnhancer
from losses  import RetinexLoss


# VIDEO_DIR  = "./Data"   # where your .mp4 files are
# FRAMES_DIR = "./frames"       # where frames will be extracted

# print(f"Extracting frames from {VIDEO_DIR}...")
# extract_frames(VIDEO_DIR, FRAMES_DIR, fps=10)   # run once to prepare data

# --- Config ---
VIDEO_DIR   = "./frames"
OUTPUT_DIR  = "results"
IMG_SIZE    = 256  
CLIP_LEN    = 5
BATCH_SIZE  = 12    # Reduced from 32 to 8 (major memory improvement)
NUM_WORKERS = 4
NOISE_CH    = 8
BASE_CH     = 32
GAMMA       = 0.4
NOISE_STD   = 0.1
EPOCHS      = 30
LR          = 1e-4
LAMBDA_REC  = 1.0
LAMBDA_IC   = 1.0
LAMBDA_REF  = 0.1
LAMBDA_SM   = 0.1
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable PyTorch memory optimization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

ckpt_dir    = Path(OUTPUT_DIR) / "checkpoints"
samples_dir = Path(OUTPUT_DIR) / "samples"
ckpt_dir.mkdir(parents=True, exist_ok=True)
samples_dir.mkdir(parents=True, exist_ok=True)


def train():
    loader = build_train_loader(
        data_root=VIDEO_DIR,
        batch_size=BATCH_SIZE,
        clip_len=CLIP_LEN,
        img_size=IMG_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = RetinexVideoEnhancer(
        noise_ch=NOISE_CH, base_ch=BASE_CH, gamma=GAMMA, noise_std=NOISE_STD,
    ).to(DEVICE)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    criterion = RetinexLoss(
        lambda_rec=LAMBDA_REC, lambda_ic=LAMBDA_IC,
        lambda_ref=LAMBDA_REF, lambda_sm=LAMBDA_SM,
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    epoch_bar = tqdm(range(EPOCHS), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        agg = {}
        n_batches = 0

        batch_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in batch_bar:
            first_frame = batch["first_frame"].to(DEVICE)

            optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with torch.amp.autocast('cuda'):
                out = model(first_frame, training_step1=True)
                loss, breakdown = criterion(
                    S=first_frame,
                    R=out["R"], I=out["I"],
                    R_t=out["R_t"], I_t=out["I_t"],
                    theta=out["theta"],
                )
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k, v in breakdown.items():
                agg[k] = agg.get(k, 0.0) + v
            n_batches += 1

            batch_bar.set_postfix(loss=f"{breakdown['loss/total']:.4f}")

        avg_loss = agg.get("loss/total", 0.0) / max(n_batches, 1)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        scheduler.step()

    torch.save({
        "epoch": EPOCHS,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, str(ckpt_dir / "final_model.pth"))

if __name__ == "__main__":
    train()