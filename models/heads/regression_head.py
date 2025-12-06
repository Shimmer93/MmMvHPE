import torch
import torch.nn as nn

from .base_head import BaseHead

class RegressionKeypointHead(BaseHead):
    def __init__(self, losses, emb_size=512, num_classes=17*3):
        super().__init__(losses)
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        # x.shape: B, M, T, N, C
        B, M, T, N, C = x.shape
        x = x[:, :, :, -1, :]  # Use the last token (skeleton token)
        x = x.mean(dim=[1,2])  # Average over modalities and temporal dimension]
        x = self.norm(x)
        x = self.fc(x)
        x = x.view(B, self.num_classes // 3, 3)
        return x
    
    def loss(self, x, data_batch):
        pred_keypoints = self.forward(x)
        
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[loss_name] = (loss_fn(pred_keypoints, data_batch['gt_keypoints']), loss_weight)
        return losses
    
    def predict(self, x):
        return self.forward(x)

        