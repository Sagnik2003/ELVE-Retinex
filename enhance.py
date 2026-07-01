"""
enhance.py — Enhance a low-light video using RetinexVideoEnhancer,
             then evaluate enhanced output vs original low-light input.
"""

import os
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
import cv2
import subprocess
import numpy as np

from model import RetinexVideoEnhancer
from metrics import evaluate_sequence

# --- Config ---
INPUT      = "./Data/P_064_V_06.mp4"
OUTPUT_DIR = "./Output"
CHECKPOINT = "./results/checkpoints/final_model.pth"
IMG_SIZE   = 256
NOISE_CH   = 8
BASE_CH    = 32
GAMMA      = 0.4
NOISE_STD  = 0.1
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Utilities ──────────────────────────────────────────────────────────────────

def ensure_dir(path: str):
    p = Path(path)
    if p.exists() and not p.is_dir():
        print(f"⚠  Removing file blocking directory creation: {p}")
        p.unlink()
    p.mkdir(parents=True, exist_ok=True)


def get_transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model():
    model = RetinexVideoEnhancer(
        noise_ch=NOISE_CH, base_ch=BASE_CH, gamma=GAMMA, noise_std=NOISE_STD
    ).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    print(f"✅ Model loaded from {CHECKPOINT} on {DEVICE}")
    return model


# ── Frame-folder mode ──────────────────────────────────────────────────────────

def enhance_frames(model):
    ensure_dir(OUTPUT_DIR)
    transform = get_transform()

    paths = sorted(
        p for p in Path(INPUT).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not paths:
        print(f"No image frames found in {INPUT}")
        return

    print(f"Found {len(paths)} frames in {INPUT}")
    enhanced_tensors  = []
    low_light_tensors = []

    with torch.no_grad():
        for i, p in enumerate(paths):
            img = transform(Image.open(p).convert("RGB")).to(DEVICE)
            low_light_tensors.append(img)

            out      = model(img.unsqueeze(0), training_step1=True)
            enhanced = (out["I"] * out["R"]).clamp(0, 1)
            enhanced_tensors.append(enhanced.squeeze(0))

            vutils.save_image(enhanced, os.path.join(OUTPUT_DIR, f"enhanced_{i:04d}.png"))

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(paths)} frames...")

    print(f"✅ All {len(paths)} enhanced frames saved to {OUTPUT_DIR}")
    _run_metrics(enhanced_tensors, low_light_tensors)


# ── Video mode ─────────────────────────────────────────────────────────────────

def enhance_video(model):
    frames_dir = os.path.join(OUTPUT_DIR, "enhanced_frames")
    ensure_dir(OUTPUT_DIR)
    ensure_dir(frames_dir)

    cap = cv2.VideoCapture(INPUT)
    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT}")
        return

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or fps > 120:
        print(f"Warning: Invalid FPS {fps}, using 30")
        fps = 30

    print(f"Video info  — FPS: {fps}, Total frames: {frame_count}")
    print(f"Device      — {DEVICE}")
    print(f"Saving enhanced frames to: {frames_dir}/n")

    transform         = get_transform()
    enhanced_tensors  = []
    low_light_tensors = []
    frame_idx         = 0

    # ── Process every frame ────────────────────────────────────────────────────
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Keep original low-light tensor for metrics
            ll_tensor = transform(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ).to(DEVICE)
            low_light_tensors.append(ll_tensor)

            # Enhance
            out_dict = model(ll_tensor.unsqueeze(0), training_step1=True)
            
            # Use model's built-in enhanced output if available, otherwise use I_gamma * R
            if "enhanced" in out_dict:
                enhanced = out_dict["enhanced"].clamp(0, 1)
            elif "I_gamma" in out_dict:
                # Use gamma-corrected illumination * reflectance
                enhanced = (out_dict["I_gamma"] * out_dict["R"]).clamp(0, 1)
            else:
                # Fallback: gamma correction on illumination
                I_gamma = torch.pow(out_dict["I"], GAMMA)
                enhanced = (I_gamma * out_dict["R"]).clamp(0, 1)
            
            enhanced_tensors.append(enhanced.squeeze(0))

            # Save frame to disk
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
            vutils.save_image(enhanced, frame_path)
            frame_idx += 1

            if frame_idx % 10 == 0:
                print(f"  Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    print(f"/n✅ All {frame_idx} frames saved to {frames_dir}")

    # ── Assemble video ─────────────────────────────────────────────────────────
    _assemble_video(frames_dir, fps)

    # ── Metrics ────────────────────────────────────────────────────────────────
    _run_metrics(enhanced_tensors, low_light_tensors)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _assemble_video(frames_dir: str, fps: float):
    output_file = os.path.join(OUTPUT_DIR, "enhanced.mp4")
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frame_files:
        print(f"⚠  No frames found in {frames_dir}")
        return

    print(f"/nAssembling {len(frame_files)} frames at {fps} FPS → {output_file}")

    # Find ffmpeg automatically or fall back to hardcoded path
    import shutil
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        # Paste your path from `where.exe ffmpeg` here
        ffmpeg_exe = r"C:/Users/Sagnik/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe"

    if not os.path.exists(ffmpeg_exe):
        print("⚠  FFmpeg not found. Run manually:")
        print(f'   ffmpeg -framerate {int(fps)} -i "{frames_dir}//frame_%05d.png" -c:v libx264 -pix_fmt yuv420p "{output_file}"')
        return

    ffmpeg_cmd = [
        ffmpeg_exe, "-y",
        "-framerate", str(int(fps)),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_file,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"✅ Enhanced video saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"⚠  FFmpeg error:/n{e.stderr}")

def _run_metrics(enhanced_tensors, low_light_tensors):
    n = min(len(enhanced_tensors), len(low_light_tensors))
    if n == 0:
        print("⚠  No frames to evaluate.")
        return

    print(f"/n📊 Evaluating metrics over {n} frames ...")
    results = evaluate_sequence(enhanced_tensors[:n], low_light_tensors[:n])

    print("/n┌─────────────────────────────────────────────────────────┐")
    print("│                  📈  Evaluation Results                 │")
    print("├──────────┬──────────────┬──────────────────────────────┤")
    print("│  Metric  │    Value     │  Interpretation              │")
    print("├──────────┼──────────────┼──────────────────────────────┤")
    print(f"│  PSNR    │ {results['psnr']:>8.4f} dB │ lower = bigger change        │")
    print(f"│  SSIM    │ {results['ssim']:>8.4f}    │ lower = more structure change│")
    print(f"│  MABD    │ {results['mabd']:>8.6f}    │ lower = less flicker         │")
    print(f"│  SPAQ    │ {str(results['spaq']):>8}       │ stub, not yet integrated     │")
    print("└──────────┴──────────────┴──────────────────────────────┘/n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = load_model()
    if Path(INPUT).is_dir():
        enhance_frames(model)
    else:
        enhance_video(model)