import torch
import torch.nn as nn
from einops import rearrange, repeat

from .base_head import BaseHead

class RegressionKeypointHeadV2(BaseHead):
    def __init__(self, losses, emb_size=512, num_joints=24, last_n_layers=-1):
        super().__init__(losses)
        self.emb_size = emb_size
        self.num_joints = num_joints
        
        self.projector = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size//2),
            nn.ReLU(),
            nn.Linear(emb_size//2, 3)
        )

        self.last_n_layers = last_n_layers

    def forward(self, x):
        print("[DEBUG]: Entered RegressionKeypointHeadV2 forward pass.")
        if isinstance(x, list):
            if self.last_n_layers > 0:
                x = x[-self.last_n_layers:]
            x = [x_[..., x_.shape[-1] // 2:] for x_ in x]
            x = torch.concatenate(x, dim=-1)
        # print(x.shape)
        # x.shape: B, M, T, J+1, C
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, M, T, N, C = x.shape

        x = x[..., N-self.num_joints:, :]
        
        # Average over temporal dimension first: B, M, J, C
        x = x.mean(dim=[1,2])
        
        # x = rearrange(x, 'b m j c -> b j (m c)')  # B, J, M*C

        x = self.projector(x)
        x = self.norm(x)
        x = self.mlp(x)
        print("[DEBUG]: RegressionKeypointHeadV2 forward pass completed.")
        return x
    
    def loss(self, x, data_batch):
        pred_keypoints = self.forward(x)
        
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[loss_name] = (loss_fn(pred_keypoints, data_batch['gt_keypoints']), loss_weight)
        # losses['attn_weights'] = (attn_weights, 0.0)  # For monitoring only
        return losses
    
    def predict(self, x):
        return self.forward(x)

        