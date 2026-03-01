from typing import Any, Dict, Optional

import torch
from torch import nn

from models.pc_encoders.mamba4d import MAMBA4DEncoder


def build_default_encoder_kwargs(encoder_dim: int) -> Dict[str, Any]:
    return {
        "radius": 0.1,
        "nsamples": 16,
        "spatial_stride": 32,
        "temporal_kernel_size": 3,
        "temporal_stride": 1,
        "emb_relu": False,
        "dim": int(encoder_dim),
        "mlp_dim": int(encoder_dim) * 2,
        "num_classes": 0,
        "depth_mamba_inter": 5,
        "rms_norm": True,
        "drop_out_in_block": 0.0,
        "drop_path": 0.1,
        "depth_mamba_intra": 1,
        "intra": True,
        "mode": "xyz",
    }


class SimpleLidarHPEModel(nn.Module):
    def __init__(
        self,
        num_joints: int = 24,
        encoder_dim: int = 512,
        head_hidden_dim: int = 1024,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        kwargs = build_default_encoder_kwargs(encoder_dim)
        if encoder_kwargs is not None:
            kwargs.update(encoder_kwargs)

        self.encoder = MAMBA4DEncoder(**kwargs)
        out_dim = int(kwargs["dim"])
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), self.num_joints * 3),
        )

    def forward(self, input_lidar: torch.Tensor) -> torch.Tensor:
        features = self.encoder(input_lidar)
        if features.dim() == 4:
            pooled = features.mean(dim=(1, 2))
        elif features.dim() == 3:
            pooled = features.mean(dim=1)
        else:
            raise ValueError(f"Unexpected encoder output shape: {tuple(features.shape)}")

        pred = self.head(pooled)
        return pred.view(pred.shape[0], self.num_joints, 3)
