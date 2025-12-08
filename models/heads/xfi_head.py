import torch
import torch.nn as nn

from .base_head import BaseHead

class regression_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=17*3):
        super(regression_Head,self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)
        self.num_joints = num_classes // 3  # Calculate number of joints from num_classes
    
    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        x = self.fc(x)
        x = x.view(x.size(0), self.num_joints, 3)  # Use dynamic num_joints
        return x

class XFiRegressionHead(BaseHead):
    def __init__(self, losses, emb_size=512, num_classes=17*3):
        super(XFiRegressionHead,self).__init__(losses)
        self.regression_head = regression_Head(emb_size, num_classes)

    def forward(self, x):
        x = self.regression_head(x) # B 17 3
        return x
    
    def loss(self, x, data_batch):
        pred_keypoints = self.forward(x)
        
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[loss_name] = (loss_fn(pred_keypoints, data_batch['gt_keypoints']), loss_weight)
        return losses
    
    def predict(self, x):
        return self.forward(x)
