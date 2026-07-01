import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import subprocess



def extract_frames(video_dir: str, frames_dir: str, fps: int = 10) -> None:
    """Extract frames from each video into a named subfolder with progress tracking."""
    videos = sorted(Path(video_dir).glob("*.mp4"))
    
    for vid_path in tqdm(videos, desc="Extracting videos", unit="video"):
        out_dir = Path(frames_dir) / vid_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cap      = cv2.VideoCapture(str(vid_path))
        src_fps  = cap.get(cv2.CAP_PROP_FPS) or fps
        interval = max(1, int(round(src_fps / fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = saved = 0
        
        pbar = tqdm(total=total_frames, desc=f"{vid_path.stem}", leave=False, unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None and idx % interval == 0:
                cv2.imwrite(str(out_dir / f"frame_{saved:05d}.png"), frame)
                saved += 1
            idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
def _collect_frame_paths(video_dir: str, exts=("*.png", "*.jpg", "*.jpeg")) -> List[str]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(paths)


def _default_transform(img_size: int = 256) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])


class LowLightVideoDataset(Dataset):
    def __init__(self, data_root, clip_len=5, img_size=256, transform=None):
        super().__init__()
        self.clip_len  = clip_len
        self.transform = transform or _default_transform(img_size)
        self.samples: List[Tuple[List[str], int]] = []

        video_dirs = sorted(d for d in Path(data_root).iterdir() if d.is_dir())
        for vdir in video_dirs:
            frames = _collect_frame_paths(str(vdir))
            for start in range(len(frames) - clip_len + 1):
                self.samples.append((frames, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, start = self.samples[idx]
        clip_paths = frame_paths[start: start + self.clip_len]
        frames = torch.stack([
            self.transform(Image.open(p).convert("RGB")) for p in clip_paths
        ])
        return {"frames": frames, "first_frame": frames[0]}


def build_train_loader(data_root, batch_size=1, clip_len=5, img_size=512, num_workers=0, shuffle=True):
    dataset = LowLightVideoDataset(data_root=data_root, clip_len=clip_len, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )