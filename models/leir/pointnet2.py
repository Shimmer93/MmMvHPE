import torch
import torch.nn as nn
from typing import Tuple

from models.pc_encoders.modules.pointnet2_modules import PointnetSAModule


class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                use_xyz=True,
            )
        )

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        x = data
        B, T, N, _ = x.shape
        x = x.reshape(-1, N, 3).float()
        with torch.cuda.amp.autocast(enabled=False):
            xyz, features = self._break_up_pc(x)
            for module in self.SA_modules:
                xyz, features = module(xyz, features)
            features = features.squeeze(-1).reshape(B, T, -1)
        return features
