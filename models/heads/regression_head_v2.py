import torch
import torch.nn as nn
from einops import rearrange, repeat

from .base_head import BaseHead

class RegressionKeypointHeadV2(BaseHead):
    def __init__(self, losses, emb_size=512):
        super().__init__(losses)
        self.emb_size = emb_size
        
        self.norm = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size//2),
            nn.ReLU(),
            nn.Linear(emb_size//2, 3)
        )

    def forward(self, x):
        print("[DEBUG]: Entered RegressionKeypointHeadV2 forward pass.")
        if isinstance(x, list):
            # print(len(x))
            x = torch.concatenate(x, dim=-1)
        # print(x.shape)
        # x.shape: B, M, T, J+1, C
        B, M, T, N, C = x.shape

        x = x[:, :, :, 1:, :]  # Use the last token (skeleton token)
        
        # Average over temporal dimension first: B, M, J, C
        x = x.mean(dim=[1,2])
        
        # x = rearrange(x, 'b m j c -> b j (m c)')  # B, J, M*C

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

        