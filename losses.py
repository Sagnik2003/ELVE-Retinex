"""
losses.py
=========
Loss functions for the Improved Retinex Low-Light Video Enhancement.

Based on the paper:
  "Efficient Low Light Video Enhancement Based on Improved Retinex Algorithms"
  Sung-Ling Lee and Shih-Hsuan Yang, ICMEW 2023

The total training loss (following RetinexDIP [2] and StableLLVE [7]) is:

    L_total = λ_rec  · L_reconstruction
            + λ_ic   · L_illumination_consistency
            + λ_ref  · L_reflection
            + λ_sm   · L_illumination_smoothness
            + λ_self · L_self_consistency      ← temporal stability term

References for individual loss terms:
  [2] Z. Zhao et al., "RetinexDIP: A unified deep framework…" 2021.
  [7] F. Zhang et al., "Learning temporal consistency…" CVPR 2021.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Individual loss components
# ============================================================================

class ReconstructionLoss(nn.Module):
    """
    L_reconstruction: element-wise reconstruction of the input image from
    the predicted Retinex components.

        L_rec = ‖ S - (R ⊙ I) ‖₁

    where S is the observed low-light image, R is the reflectance,
    and I is the illumination map.
    """

    def forward(
        self,
        S: torch.Tensor,   # (B, 3, H, W) input
        R: torch.Tensor,   # (B, 3, H, W) reflectance
        I: torch.Tensor,   # (B, 1, H, W) illumination (broadcast over channels)
    ) -> torch.Tensor:
        reconstructed = R * I          # element-wise (I is broadcast)
        return F.l1_loss(reconstructed, S)


class IlluminationConsistencyLoss(nn.Module):
    """
    L_illumination_consistency: enforces that the illumination maps
    produced from the original and the affine-transformed frames are
    geometrically consistent after inverse-warping.

        L_ic = ‖ I - warp(I_t, θ⁻¹) ‖₁

    In practice we use the forward warp and compare directly:
        L_ic = ‖ warp(I, θ) - I_t ‖₁

    This is the *self-consistency loss* described in Figure 2 of the paper.
    """

    def forward(
        self,
        I:     torch.Tensor,  # (B, 1, H, W) illumination from original frame
        I_t:   torch.Tensor,  # (B, 1, H, W) illumination from transformed frame
        theta: torch.Tensor,  # (B, 2, 3)    affine parameters used in warp
    ) -> torch.Tensor:
        grid   = F.affine_grid(theta, I.size(), align_corners=False)
        I_warped = F.grid_sample(I, grid, mode="bilinear",
                                 padding_mode="border", align_corners=False)
        return F.l1_loss(I_warped, I_t)


class ReflectionLoss(nn.Module):
    """
    L_reflection: smoothness prior on the reflectance map.

    Natural reflectance maps should be piecewise smooth, so we penalise
    large gradients in R while preserving edges (weighted by the image gradient).

        L_ref = ‖ ∇R ⊙ exp(-λ · ∇S) ‖₁

    Args:
        edge_weight : λ for the edge-awareness term (default 1.0).
    """

    def __init__(self, edge_weight: float = 1.0) -> None:
        super().__init__()
        self.edge_weight = edge_weight

    @staticmethod
    def _gradient(x: torch.Tensor) -> Tuple_[torch.Tensor, torch.Tensor]:
        """Compute horizontal and vertical image gradients."""
        dx = x[:, :, :, :-1] - x[:, :, :, 1:]   # (B, C, H, W-1)
        dy = x[:, :, :-1, :] - x[:, :, 1:, :]   # (B, C, H-1, W)
        return dx, dy

    def forward(
        self,
        R: torch.Tensor,   # (B, 3, H, W)
        S: torch.Tensor,   # (B, 3, H, W)
    ) -> torch.Tensor:
        R_dx, R_dy = self._gradient(R)
        S_dx, S_dy = self._gradient(S)

        # Edge-aware weights
        w_dx = torch.exp(-self.edge_weight * S_dx.abs().mean(dim=1, keepdim=True))
        w_dy = torch.exp(-self.edge_weight * S_dy.abs().mean(dim=1, keepdim=True))

        loss = (R_dx.abs() * w_dx).mean() + (R_dy.abs() * w_dy).mean()
        return loss


# small alias to avoid importing Tuple from typing inside the function body
from typing import Tuple as Tuple_


class IlluminationSmoothnessLoss(nn.Module):
    """
    L_illumination_smoothness: the illumination map should vary smoothly
    to avoid amplifying noise in dark regions.

        L_sm = ‖ ∇I ‖₁
    """

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        dx = I[:, :, :, :-1] - I[:, :, :, 1:]
        dy = I[:, :, :-1, :] - I[:, :, 1:, :]
        return dx.abs().mean() + dy.abs().mean()


# ============================================================================
# Combined loss
# ============================================================================

class RetinexLoss(nn.Module):
    """
    Combines all four loss terms into the total training objective.

    L_total = λ_rec  · L_reconstruction
            + λ_ic   · L_illumination_consistency
            + λ_ref  · L_reflection
            + λ_sm   · L_illumination_smoothness

    Default weights match typical RetinexDIP settings and give a good
    starting point; tune them based on your dataset.

    Args:
        lambda_rec  : Weight for reconstruction loss    (default 1.0).
        lambda_ic   : Weight for illumination consistency (default 1.0).
        lambda_ref  : Weight for reflection smoothness  (default 0.1).
        lambda_sm   : Weight for illumination smoothness (default 0.1).
        edge_weight : Edge-awareness coefficient in reflection loss.
    """

    def __init__(
        self,
        lambda_rec:  float = 1.0,
        lambda_ic:   float = 1.0,
        lambda_ref:  float = 0.1,
        lambda_sm:   float = 0.1,
        edge_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_ic  = lambda_ic
        self.lambda_ref = lambda_ref
        self.lambda_sm  = lambda_sm

        self.l_rec = ReconstructionLoss()
        self.l_ic  = IlluminationConsistencyLoss()
        self.l_ref = ReflectionLoss(edge_weight=edge_weight)
        self.l_sm  = IlluminationSmoothnessLoss()

    def forward(
        self,
        S:     torch.Tensor,   # (B, 3, H, W) input low-light frame
        R:     torch.Tensor,   # (B, 3, H, W) predicted reflectance
        I:     torch.Tensor,   # (B, 1, H, W) predicted illumination
        R_t:   torch.Tensor,   # (B, 3, H, W) reflectance of transformed frame
        I_t:   torch.Tensor,   # (B, 1, H, W) illumination of transformed frame
        theta: torch.Tensor,   # (B, 2, 3)    affine warp parameters
    ) -> Tuple_[torch.Tensor, dict]:
        """
        Args:
            S, R, I     : Outputs for the original frame.
            R_t, I_t    : Outputs for the affine-transformed frame.
            theta       : Affine parameters predicted by AffineWarp.

        Returns:
            total_loss : Scalar loss tensor (differentiable).
            breakdown  : Dict of individual weighted loss values (for logging).
        """
        l_rec = self.l_rec(S, R, I)
        l_ic  = self.l_ic (I, I_t, theta)
        l_ref = self.l_ref(R, S)
        l_sm  = self.l_sm (I)

        total = (
            self.lambda_rec * l_rec
            + self.lambda_ic  * l_ic
            + self.lambda_ref * l_ref
            + self.lambda_sm  * l_sm
        )

        breakdown = {
            "loss/reconstruction"           : l_rec.item(),
            "loss/illumination_consistency" : l_ic.item(),
            "loss/reflection_smoothness"    : l_ref.item(),
            "loss/illumination_smoothness"  : l_sm.item(),
            "loss/total"                    : total.item(),
        }
        return total, breakdown
