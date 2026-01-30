import torch
import torch.nn as nn


class PoseIdentityBackbone(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        if x is None:
            return None
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"PoseIdentityBackbone expected input_dim={self.input_dim}, got {x.shape[-1]}")
        return x


class Pose2DBackbone(PoseIdentityBackbone):
    def __init__(self):
        super().__init__(input_dim=2)


class Pose3DBackbone(PoseIdentityBackbone):
    def __init__(self):
        super().__init__(input_dim=3)
