"""
model.py
========
Neural network components for the Improved Retinex Low-Light Video Enhancement.

Based on the paper:
  "Efficient Low Light Video Enhancement Based on Improved Retinex Algorithms"
  Sung-Ling Lee and Shih-Hsuan Yang, ICMEW 2023

Architecture overview (Section 2 of the paper):
─────────────────────────────────────────────────────────────────────────────
  Step 1  (per-video, first frame only):
    • VGG19 shadow features  ─┐
    • Random noise            ├─► DIP Network ──► Reflectance + Illumination
    • First frame             ─┘
      ↓  (affine-transformed copy) ──► DIP Network (shared weights) ──► consistency loss

  Step 2  (remaining frames in sequence):
    • Remaining frames ──► Trained DIP Network ──► Illumination
    • Gamma-corrected Illumination ──► element-wise divide ──► Enhanced frame
─────────────────────────────────────────────────────────────────────────────

Key design choices:
  1. VGG19 first conv-block features (edges + colour) augment the DIP input.
  2. Deformable convolution kernels replace standard fixed kernels in the DIP.
  3. Object-based affine optical-flow for temporal consistency.
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ============================================================================
# 1.  VGG19 Shadow Feature Extractor
# ============================================================================

class VGGShadowFeatures(nn.Module):
    """
    Extracts low-level (shadow) features from the *first convolution block*
    of a pre-trained VGG-19 network (i.e., the first 4 layers).

    The paper states these features encode edges and colour, which stabilise
    the DIP training and reduce the number of iterations required.

    Args:
        freeze : If True (default), the VGG weights are frozen during training.
    """

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1)
        # First conv block = features[0:4]  (Conv-ReLU-Conv-ReLU)
        self.block1 = nn.Sequential(*list(vgg.features.children())[:4])

        if freeze:
            for p in self.block1.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)  input image in [0, 1]
        Returns:
            feats : (B, 64, H, W)  VGG shadow features
        """
        return self.block1(x)


# ============================================================================
# 2.  Deformable Convolution (lightweight 2-D implementation)
# ============================================================================

class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v1 (Dai et al., ICCV 2017).

    A random trainable offset is added to the sampling grid, allowing the
    kernel to adapt to translation, scale, aspect-ratio, and rotation –
    as described in Section 2.2 of the paper.

    Uses torch's built-in ``grid_sample`` for differentiable sampling;
    this avoids the need for custom CUDA extensions.

    Args:
        in_channels  : Number of input channels.
        out_channels : Number of output channels.
        kernel_size  : Convolution kernel size (square).
        stride       : Convolution stride.
        padding      : Zero-padding added to both sides.
        bias         : Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        padding:      int = 1,
        bias:         bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

        # Offset predictor: outputs 2 * k^2 channels (dx, dy per kernel cell)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias,   0)

        # Main weight applied after deformable sampling
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias_param = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    # ------------------------------------------------------------------
    def _get_deformed_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply offset to a regular grid and bilinearly sample from *x*.
        Returns a tensor of shape (B, C*k*k, H_out, W_out) suitable for
        a follow-up F.conv2d with groups=B or unfolding.
        """
        B, C, H, W = x.shape
        k = self.kernel_size

        # Predict offsets  → (B, 2*k*k, H, W)
        offsets = self.offset_conv(x)

        # Build a regular sampling grid  → (B, k*k, H, W, 2)
        # Each kernel position defines a base offset from the center pixel
        ky = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        kx = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")  # (k, k)
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, k * k, 1, 1, 2)
        base_grid = base_grid.expand(B, -1, H, W, -1)           # (B, k*k, H, W, 2)

        # Add learned offsets
        offsets = offsets.view(B, k * k, 2, H, W).permute(0, 1, 3, 4, 2)  # (B,k*k,H,W,2)
        sample_grid = base_grid + offsets                                   # (B,k*k,H,W,2)

        # Normalise to [-1, 1] for grid_sample
        norm_x = sample_grid[..., 0] / (W / 2) + torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W) * 2 / W - 1
        norm_y = sample_grid[..., 1] / (H / 2) + torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1) * 2 / H - 1

        # Clamp to valid range
        norm_x = norm_x.clamp(-1, 1)
        norm_y = norm_y.clamp(-1, 1)
        grid_norm = torch.stack([norm_x, norm_y], dim=-1)  # (B, k*k, H, W, 2)

        # Sample channel-by-channel then concat  → (B, C*k*k, H, W)
        sampled_list = []
        for ki in range(k * k):
            g = grid_norm[:, ki, :, :, :]  # (B, H, W, 2)
            # grid_sample expects (B,C,H_in,W_in) and grid (B,H_out,W_out,2)
            s = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled_list.append(s)          # each (B, C, H, W)

        return torch.cat(sampled_list, dim=1)  # (B, C*k*k, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        deformed = self._get_deformed_features(x)  # (B, C*k^2, H, W)

        # Reshape weight to (out, in*k^2, 1, 1) so conv2d acts as 1×1
        w = self.weight.view(self.weight.size(0), -1, 1, 1)
        out = F.conv2d(deformed, w, bias=self.bias_param)
        return out


# ============================================================================
# 3.  Deep Image Prior (DIP) Generator
# ============================================================================

class DIPNetwork(nn.Module):
    """
    U-Net style Deep Image Prior network used as the Retinex generator.

    Takes as input a concatenation of:
      • The low-light image          (3 channels)
      • VGG-19 shadow features       (64 channels)
      • Random noise                 (noise_ch channels)
    Total input channels = 3 + 64 + noise_ch

    Produces two separate outputs via two lightweight decoder heads:
      • Reflectance map   R  in [0, 1]   (3 channels)
      • Illumination map  I  in [0, 1]   (1 channel)

    Key modification over vanilla DIP: deformable convolution in encoder
    blocks for better spatial adaptivity (Section 2.2 of the paper).

    Args:
        noise_ch   : Channels of the random noise input (default 8).
        base_ch    : Base channel multiplier for the encoder/decoder (default 32).
    """

    def __init__(self, noise_ch: int = 8, base_ch: int = 32) -> None:
        super().__init__()
        in_ch = 3 + 64 + noise_ch   # image + VGG feats + noise

        # ------ Encoder ------
        self.enc1 = self._make_block(in_ch,       base_ch,     deformable=True)
        self.enc2 = self._make_block(base_ch,     base_ch * 2, deformable=True)
        self.enc3 = self._make_block(base_ch * 2, base_ch * 4, deformable=True)
        self.pool  = nn.MaxPool2d(2)

        # ------ Bottleneck ------
        self.bottleneck = self._make_block(base_ch * 4, base_ch * 8)

        # ------ Decoder ------
        self.up3   = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3  = self._make_block(base_ch * 8, base_ch * 4)   # +skip
        self.up2   = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2  = self._make_block(base_ch * 4, base_ch * 2)
        self.up1   = nn.ConvTranspose2d(base_ch * 2, base_ch,     2, stride=2)
        self.dec1  = self._make_block(base_ch * 2, base_ch)

        # ------ Output heads ------
        self.head_R = nn.Sequential(
            nn.Conv2d(base_ch, 3, 1), nn.Sigmoid()
        )
        self.head_I = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1), nn.Sigmoid()
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _make_block(
        in_ch: int, out_ch: int, deformable: bool = False
    ) -> nn.Sequential:
        """Two-conv residual-style block, optionally using DeformableConv2d."""
        conv_cls = DeformableConv2d if deformable else nn.Conv2d
        return nn.Sequential(
            conv_cls(in_ch,  out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            conv_cls(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    # ------------------------------------------------------------------
    def forward(
        self, img: torch.Tensor, vgg_feats: torch.Tensor, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img       : (B, 3,  H, W) low-light image in [0, 1]
            vgg_feats : (B, 64, H, W) VGG-19 block-1 features
            noise     : (B, noise_ch, H, W) random Gaussian noise

        Returns:
            R : (B, 3, H, W) predicted reflectance in [0, 1]
            I : (B, 1, H, W) predicted illumination in [0, 1]
        """
        x = torch.cat([img, vgg_feats, noise], dim=1)

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b  = self.bottleneck(self.pool(e3))

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        R = self.head_R(d1)   # reflectance
        I = self.head_I(d1)   # illumination
        return R, I


# ============================================================================
# 4.  Affine Warp Utility (optical-flow approximation)
# ============================================================================

class AffineWarp(nn.Module):
    """
    Approximates the optical flow between consecutive frames as a constrained
    affine transformation (Section 2.3 of the paper).

    A lightweight CNN predicts 6 affine parameters (θ) from a pair of frames,
    and the reference frame is warped using ``F.affine_grid`` + ``F.grid_sample``.

    This replaces expensive per-pixel optical-flow estimation with a low-cost
    global affine, which can be applied repeatedly for improved accuracy.
    """

    def __init__(self, in_ch: int = 6) -> None:
        super().__init__()
        # Simple localisation network: 2-frame input (6 channels = 2 × RGB)
        self.localizer = nn.Sequential(
            nn.Conv2d(in_ch, 32, 7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(32,    64, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )
        # Initialise to identity transform
        nn.init.zeros_(self.localizer[-1].weight)
        self.localizer[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src : (B, 3, H, W) source frame (frame t)
            tgt : (B, 3, H, W) target frame  (frame t+1)

        Returns:
            warped : (B, 3, H, W) src warped to align with tgt
            theta  : (B, 2, 3)   affine parameters
        """
        pair  = torch.cat([src, tgt], dim=1)  # (B, 6, H, W)
        theta = self.localizer(pair).view(-1, 2, 3)
        grid  = F.affine_grid(theta, src.size(), align_corners=False)
        warped = F.grid_sample(src, grid, mode="bilinear",
                               padding_mode="border", align_corners=False)
        return warped, theta


# ============================================================================
# 5.  Full Retinex Video Enhancement Model
# ============================================================================

class RetinexVideoEnhancer(nn.Module):
    """
    Assembles all components into the full video enhancement pipeline.

    Pipeline (matches Figure 2 of the paper):
    ──────────────────────────────────────────
    Training (first frame):
      1. Extract VGG-19 block-1 features from the first frame.
      2. Generate Retinex components (R, I) via DIPNetwork.
      3. Apply a random affine transform to the first frame.
      4. Pass the transformed frame through the *same* DIPNetwork (shared weights).
      5. Compute self-consistency loss between the two sets of outputs.

    Inference (subsequent frames):
      1. Feed each frame through the frozen DIPNetwork → illumination I.
      2. Apply Gamma correction to I.
      3. Enhanced frame = original_frame / (gamma_corrected_I + ε).
    ──────────────────────────────────────────

    Args:
        noise_ch : Channels for the random noise input.
        base_ch  : Base channel width of DIPNetwork.
        gamma    : Gamma-correction exponent for illumination (default 0.4 ≈ 1/2.5).
        noise_std: Standard deviation of the random noise.
    """

    def __init__(
        self,
        noise_ch:  int   = 8,
        base_ch:   int   = 32,
        gamma:     float = 0.4,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.noise_ch  = noise_ch
        self.noise_std = noise_std

        self.vgg_extractor = VGGShadowFeatures(freeze=True)
        self.dip_network   = DIPNetwork(noise_ch=noise_ch, base_ch=base_ch)
        self.affine_warp   = AffineWarp()

    # ------------------------------------------------------------------
    def _sample_noise(self, like: torch.Tensor) -> torch.Tensor:
        """Return random Gaussian noise with the same (B, noise_ch, H, W) shape."""
        B, _, H, W = like.shape
        return torch.randn(B, self.noise_ch, H, W, device=like.device) * self.noise_std

    # ------------------------------------------------------------------
    def forward(
        self, frame: torch.Tensor, training_step1: bool = True
    ) -> dict:
        """
        Single-frame forward pass used during *Step 1* (first-frame training).

        Args:
            frame         : (B, 3, H, W) low-light frame in [0, 1].
            training_step1: If True, also computes the consistency branch.

        Returns a dict with:
            'R'          : (B, 3, H, W) reflectance
            'I'          : (B, 1, H, W) illumination
            'R_t'        : (B, 3, H, W) reflectance of affine-transformed frame
            'I_t'        : (B, 1, H, W) illumination of affine-transformed frame
            'theta'      : (B, 2, 3)    affine parameters used
            'enhanced'   : (B, 3, H, W) gamma-corrected output
        """
        vgg_feats = self.vgg_extractor(frame)   # (B, 64, H, W)
        noise     = self._sample_noise(frame)

        R, I = self.dip_network(frame, vgg_feats, noise)

        out: dict = {"R": R, "I": I}

        if training_step1:
            # Random affine transform for self-consistency (Section 2.3)
            dummy_tgt = frame + 0.0   # use same frame as target; transformation is random
            transformed, theta = self.affine_warp(frame, dummy_tgt)

            # Transform VGG features and noise to match
            vgg_t = self.vgg_extractor(transformed)
            noise_t = self._sample_noise(transformed)

            R_t, I_t = self.dip_network(transformed, vgg_t, noise_t)
            out["R_t"]   = R_t
            out["I_t"]   = I_t
            out["theta"] = theta

        # Enhanced frame: frame / (gamma-corrected I + eps)
        I_gamma   = torch.pow(I + 1e-6, self.gamma)
        enhanced  = frame / (I_gamma + 1e-6)
        enhanced  = enhanced.clamp(0, 1)
        out["enhanced"] = enhanced
        out["I_gamma"]  = I_gamma
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def enhance_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Inference-mode enhancement for a full clip *after* Step 1 training.

        The DIPNetwork must already be trained on the first frame before
        calling this.

        Args:
            frames : (T, 3, H, W) full sequence of low-light frames in [0,1].
                     (No batch dimension – process one video at a time.)

        Returns:
            enhanced : (T, 3, H, W) enhanced sequence in [0, 1].
        """
        self.eval()
        results = []
        for t in range(frames.size(0)):
            f = frames[t].unsqueeze(0)            # (1, 3, H, W)
            vgg_f = self.vgg_extractor(f)
            noise  = self._sample_noise(f)
            _, I   = self.dip_network(f, vgg_f, noise)
            I_gamma = torch.pow(I + 1e-6, self.gamma)
            enh = (f / (I_gamma + 1e-6)).clamp(0, 1)
            results.append(enh.squeeze(0))        # (3, H, W)
        return torch.stack(results)               # (T, 3, H, W)
