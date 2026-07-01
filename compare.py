"""
compare.py
==========
Compare enhanced video with original low-light video using metrics.

Usage:
    python compare.py
"""

import os
from pathlib import Path
import json

import torch
import cv2
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np

from metrics import psnr, ssim, mabd

# --- Config ---
INPUT_VIDEO    = "./Data/P_064_V_06.mp4"
ENHANCED_FRAMES_DIR = "./Output/enhanced_frames"
OUTPUT_DIR     = "./Output"
IMG_SIZE       = 256
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])


def load_video_frames(video_path, max_frames=None):
    """Load frames from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    frames = []
    transform = get_transform()
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB and preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(Image.fromarray(frame_rgb)).to(DEVICE)
        frames.append(frame_tensor)
        frame_idx += 1
        
        if max_frames and frame_idx >= max_frames:
            break
    
    cap.release()
    return frames


def load_enhanced_frames(frames_dir, max_frames=None):
    """Load enhanced frames from directory."""
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        print(f"Error: Enhanced frames directory not found: {frames_dir}")
        return []
    
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    frames = []
    transform = get_transform()
    
    for i, frame_path in enumerate(frame_files):
        if max_frames and i >= max_frames:
            break
        frame_tensor = transform(Image.open(frame_path).convert("RGB")).to(DEVICE)
        frames.append(frame_tensor)
    
    return frames


def compare_videos():
    """Compare enhanced video with original."""
    print("Loading original video frames...")
    original_frames = load_video_frames(INPUT_VIDEO)
    
    print("Loading enhanced frames...")
    enhanced_frames = load_enhanced_frames(ENHANCED_FRAMES_DIR)
    
    if not original_frames:
        print("Error: No original frames loaded")
        return
    
    if not enhanced_frames:
        print("Error: No enhanced frames loaded")
        return
    
    # Ensure same number of frames
    min_frames = min(len(original_frames), len(enhanced_frames))
    original_frames = original_frames[:min_frames]
    enhanced_frames = enhanced_frames[:min_frames]
    
    print(f"\nComparing {min_frames} frames...")
    
    # Compute metrics for each frame
    psnr_values = []
    ssim_values = []
    mabd_values = []
    
    for i in tqdm(range(min_frames), desc="Computing metrics"):
        orig = original_frames[i].unsqueeze(0)
        enh = enhanced_frames[i].unsqueeze(0)
        
        psnr_val = psnr(enh, orig)
        ssim_val = ssim(enh, orig)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    
    # Compute MABD (temporal flickering)
    mabd_val = mabd(enhanced_frames)
    
    # Compute statistics
    psnr_mean = np.mean(psnr_values)
    psnr_std = np.std(psnr_values)
    ssim_mean = np.mean(ssim_values)
    ssim_std = np.std(ssim_values)
    
    # Results summary
    results = {
        "total_frames": min_frames,
        "metrics": {
            "psnr": {
                "mean": float(psnr_mean),
                "std": float(psnr_std),
                "min": float(np.min(psnr_values)),
                "max": float(np.max(psnr_values)),
            },
            "ssim": {
                "mean": float(ssim_mean),
                "std": float(ssim_std),
                "min": float(np.min(ssim_values)),
                "max": float(np.max(ssim_values)),
            },
            "mabd": float(mabd_val),
        }
    }
    
    # Print results
    print("\n" + "="*70)
    print("COMPARISON RESULTS: Enhanced vs Original")
    print("="*70)
    print(f"Total frames analyzed: {min_frames}")
    print()
    print("PSNR (Peak Signal-to-Noise Ratio):")
    print(f"  Mean:  {psnr_mean:.2f} dB")
    print(f"  Std:   {psnr_std:.2f} dB")
    print(f"  Range: {np.min(psnr_values):.2f} - {np.max(psnr_values):.2f} dB")
    print()
    print("SSIM (Structural Similarity Index):")
    print(f"  Mean:  {ssim_mean:.4f}")
    print(f"  Std:   {ssim_std:.4f}")
    print(f"  Range: {np.min(ssim_values):.4f} - {np.max(ssim_values):.4f}")
    print()
    print("MABD (Mean Absolute Brightness Difference - Temporal Flicker):")
    print(f"  Value: {mabd_val:.4f}")
    print(f"  (Lower is better - less temporal flicker)")
    print()
    print("INTERPRETATION:")
    print(f"  • Low PSNR/SSIM = Strong enhancement (more change from original)")
    print(f"  • High PSNR/SSIM = Subtle enhancement (closer to original)")
    print(f"  • Low MABD = Smooth temporal consistency (good)")
    print("="*70)
    
    # Save results to JSON
    results_file = Path(OUTPUT_DIR) / "comparison_metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    compare_videos()
