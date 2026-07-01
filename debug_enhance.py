"""
debug_enhance.py
================
Debug script to check model outputs and diagnose enhancement issues.
"""

import torch
import cv2
import torchvision.transforms as T
from PIL import Image
import torchvision.utils as vutils
import os
from pathlib import Path

from model import RetinexVideoEnhancer

# --- Config ---
INPUT_VIDEO   = "./Data/P_064_V_06.mp4"
OUTPUT_DIR    = "./Output/debug"
CHECKPOINT    = "./results/checkpoints/final_model.pth"
IMG_SIZE      = 256
NOISE_CH      = 8
BASE_CH       = 32
GAMMA         = 0.4
NOISE_STD     = 0.1
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    model = RetinexVideoEnhancer(
        noise_ch=NOISE_CH, base_ch=BASE_CH, gamma=GAMMA, noise_std=NOISE_STD
    ).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    return model


def get_transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])


def debug_enhance():
    print("Loading model...")
    model = load_model()
    
    print(f"Loading video from {INPUT_VIDEO}...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video")
        return
    
    transform = get_transform()
    
    # Process first frame only for debugging
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    
    cap.release()
    
    # Preprocess
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_light = transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)
    
    # Save original
    vutils.save_image(low_light, os.path.join(OUTPUT_DIR, "00_original.png"))
    print(f"✓ Saved original frame")
    
    # Get model output
    print("\nRunning model inference...")
    with torch.no_grad():
        out = model(low_light, training_step1=True)
    
    # Analyze outputs
    print("\n" + "="*70)
    print("MODEL OUTPUT ANALYSIS")
    print("="*70)
    
    print(f"\nOutput keys: {out.keys()}")
    
    for key in ["I", "R", "I_t", "R_t", "I_gamma", "enhanced"]:
        if key in out:
            tensor = out[key]
            print(f"\n{key}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Min: {tensor.min().item():.6f}")
            print(f"  Max: {tensor.max().item():.6f}")
            print(f"  Mean: {tensor.mean().item():.6f}")
            print(f"  Std: {tensor.std().item():.6f}")
    
    # Try different enhancement combinations
    print("\n" + "="*70)
    print("TESTING ENHANCEMENT COMBINATIONS")
    print("="*70)
    
    I = out["I"]
    R = out["R"]
    
    # Method 1: Direct multiplication (current)
    enhanced1 = (I * R).clamp(0, 1)
    vutils.save_image(enhanced1, os.path.join(OUTPUT_DIR, "01_I_times_R.png"))
    print("\n✓ Method 1: I * R (current method)")
    print(f"  Mean: {enhanced1.mean().item():.6f}")
    
    # Method 2: With gamma enhancement on I
    I_gamma = torch.pow(I, GAMMA)  # Apply gamma correction
    enhanced2 = (I_gamma * R).clamp(0, 1)
    vutils.save_image(enhanced2, os.path.join(OUTPUT_DIR, "02_I_gamma_times_R.png"))
    print(f"\n✓ Method 2: I^{GAMMA} * R (with gamma enhancement)")
    print(f"  Mean: {enhanced2.mean().item():.6f}")
    
    # Method 3: Stronger gamma
    I_gamma_strong = torch.pow(I, 0.5)  # Stronger enhancement
    enhanced3 = (I_gamma_strong * R).clamp(0, 1)
    vutils.save_image(enhanced3, os.path.join(OUTPUT_DIR, "03_I_gamma_0.5_times_R.png"))
    print(f"\n✓ Method 3: I^0.5 * R (stronger gamma)")
    print(f"  Mean: {enhanced3.mean().item():.6f}")
    
    # Method 4: Just illumination
    enhanced4 = I.clamp(0, 1)
    vutils.save_image(enhanced4, os.path.join(OUTPUT_DIR, "04_I_only.png"))
    print(f"\n✓ Method 4: I only (illumination)")
    print(f"  Mean: {enhanced4.mean().item():.6f}")
    
    # Method 5: Just reflectance
    enhanced5 = R.clamp(0, 1)
    vutils.save_image(enhanced5, os.path.join(OUTPUT_DIR, "05_R_only.png"))
    print(f"\n✓ Method 5: R only (reflectance)")
    print(f"  Mean: {enhanced5.mean().item():.6f}")
    
    # Method 6: Average of all enhancements
    enhanced6 = ((enhanced1 + enhanced2 + enhanced3) / 3).clamp(0, 1)
    vutils.save_image(enhanced6, os.path.join(OUTPUT_DIR, "06_average_all.png"))
    print(f"\n✓ Method 6: Average of methods 1-3")
    print(f"  Mean: {enhanced6.mean().item():.6f}")
    
    # Method 7: Model's pre-computed enhanced output (if available)
    if "enhanced" in out:
        enhanced7 = out["enhanced"].clamp(0, 1)
        vutils.save_image(enhanced7, os.path.join(OUTPUT_DIR, "07_model_enhanced.png"))
        print(f"\n✓ Method 7: Model's built-in 'enhanced' output")
        print(f"  Mean: {enhanced7.mean().item():.6f}")
    
    # Method 8: Model's pre-computed I_gamma (if available)
    if "I_gamma" in out:
        enhanced8 = (out["I_gamma"] * R).clamp(0, 1)
        vutils.save_image(enhanced8, os.path.join(OUTPUT_DIR, "08_I_gamma_times_R.png"))
        print(f"\n✓ Method 8: Model's I_gamma * R")
        print(f"  Mean: {enhanced8.mean().item():.6f}")
    
    print("\n" + "="*70)
    print(f"All debug images saved to: {OUTPUT_DIR}")
    print("\nCompare these images to find the best enhancement method:")
    print("  00_original.png - Original low-light image")
    print("  01_I_times_R.png - Current method")
    print("  02_I_gamma_times_R.png - With gamma on I")
    print("  03_I_gamma_0.5_times_R.png - Stronger gamma")
    print("  04_I_only.png - Illumination only")
    print("  05_R_only.png - Reflectance only")
    print("  06_average_all.png - Average of best methods")
    print("="*70)


if __name__ == "__main__":
    debug_enhance()
