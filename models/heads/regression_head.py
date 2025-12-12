import torch
import torch.nn as nn
import torch.nn.functional as F

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


class RegressionCameraHead(BaseHead):
    """Head for predicting camera parameters from camera tokens.
    
    Predicts camera pose encoding with 9 parameters:
    - [:3] = absolute translation vector T (3D)
    - [3:7] = rotation as quaternion quat (4D)
    - [7:] = field of view (2D)
    """
    def __init__(self, losses, emb_size=512, num_params=9, 
                 weight_trans=1.0, weight_rot=1.0, weight_focal=0.5):
        super().__init__(losses)
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_params)
        self.num_params = num_params
        
        # Loss weights for different components
        self.weight_trans = weight_trans
        self.weight_rot = weight_rot
        self.weight_focal = weight_focal

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        # x.shape: B, M, T, N, C
        # N contains: [camera_token, joint_token, register_tokens..., patch_tokens...]
        B, M, T, N, C = x.shape
        x = x[:, :, :, 0, :]  # Use the first token (camera token)
        x = x.mean(dim=[1, 2])  # Average over modalities and temporal dimension
        x = self.norm(x)
        x = self.fc(x)
        # x.shape: B, 9 (3 for translation, 4 for quaternion, 2 for FoV)
        return x
    
    def loss(self, x, data_batch):
        pred_camera = self.forward(x)  # B, 9
        
        # Get ground truth camera parameters for the anchor modality
        anchor_key = data_batch.get('anchor_key', 'input_rgb')
        if isinstance(anchor_key, (list, tuple)):
            anchor_key = anchor_key[0]
        
        # Extract modality name from anchor_key (e.g., 'input_rgb' -> 'rgb')
        modality = anchor_key.replace('input_', '')
        gt_key = f'gt_camera_{modality}'
        
        if gt_key not in data_batch:
            raise KeyError(f"Ground truth camera parameters '{gt_key}' not found in data_batch. "
                         f"Available keys: {list(data_batch.keys())}")
        
        gt_camera = data_batch[gt_key]  # B, T, 9
        
        # Average ground truth over temporal dimension to match prediction
        gt_camera = gt_camera.mean(dim=1)  # B, 9
        
        # Split into components: translation (3), rotation (4), focal (2)
        pred_trans = pred_camera[:, :3]
        pred_rot = pred_camera[:, 3:7]
        pred_focal = pred_camera[:, 7:9]
        
        gt_trans = gt_camera[:, :3]
        gt_rot = gt_camera[:, 3:7]
        gt_focal = gt_camera[:, 7:9]
        
        # Compute L1 losses for each component
        loss_trans = F.l1_loss(pred_trans, gt_trans)
        loss_rot = F.l1_loss(pred_rot, gt_rot)
        loss_focal = F.l1_loss(pred_focal, gt_focal)
        
        # Combine weighted losses
        total_loss = (self.weight_trans * loss_trans + 
                     self.weight_rot * loss_rot + 
                     self.weight_focal * loss_focal)
        
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            # Use the computed total_loss and apply the configured loss_weight
            losses[loss_name] = (total_loss, loss_weight)
        
        # Also return individual component losses for monitoring
        losses['camera_trans'] = (loss_trans, self.weight_trans)
        losses['camera_rot'] = (loss_rot, self.weight_rot)
        losses['camera_focal'] = (loss_focal, self.weight_focal)
        
        return losses
    
    def predict(self, x):
        return self.forward(x)


