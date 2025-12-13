import torch
import torch.nn as nn

from .base_head import BaseHead

class RegressionKeypointHeadV2(BaseHead):
    def __init__(self, losses, emb_size=512, num_modalities=4):
        super().__init__(losses)
        self.emb_size = emb_size
        self.num_modalities = num_modalities
        
        # Modality attention mechanism
        self.modality_attention = nn.Sequential(
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, 1)
        )
        
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, 3)

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        # x.shape: B, M, T, J+1, C
        B, M, T, N, C = x.shape

        x = x[:, :, :, 1:, :]  # Use the last token (skeleton token)
        
        # Average over temporal dimension first: B, M, J, C
        x_temporal = x.mean(dim=2)
        
        # Average over joints for modality attention: B, M, C
        x_mod = x_temporal.mean(dim=2)
        
        # Compute attention scores for each modality: B, M, 1
        attn_scores = self.modality_attention(x_mod)  # B, M, 1
        attn_weights = torch.softmax(attn_scores, dim=1)  # B, M, 1
        
        # Apply attention weights: B, M, J, C -> B, J, C
        attn_weights = attn_weights.unsqueeze(2)  # B, M, 1, 1
        x_weighted = (x_temporal * attn_weights).sum(dim=1)  # B, J, C
        
        x = self.norm(x_weighted)
        x = self.fc(x)
        return x, attn_weights.squeeze(-1)
    
    def loss(self, x, data_batch):
        pred_keypoints, attn_weights = self.forward(x)
        
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[loss_name] = (loss_fn(pred_keypoints, data_batch['gt_keypoints']), loss_weight)
        losses['attn_weights'] = (attn_weights, 0.0)  # For monitoring only
        return losses
    
    def predict(self, x):
        return self.forward(x)[0]

        