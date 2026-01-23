"""
SMPL Head for predicting SMPL parameters from aggregated features.

This head predicts SMPL parameters only:
- global_orient: (B, 3) - root rotation in axis-angle
- body_pose: (B, 69) - body joint rotations (23 joints × 3)
- betas: (B, 10) - shape parameters
- transl: (B, 3) - translation

Note: Conversion to keypoints/vertices is handled by metrics and visualization,
not by this head. This keeps the head lightweight and separates concerns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_head import BaseHead


class SMPLHead(BaseHead):
    """
    SMPL parameter regression head.
    
    Predicts SMPL parameters (pose, shape, translation) from aggregated features
    using a simple MLP with activation functions.
    
    Args:
        losses: List of loss configurations
        emb_size: Input embedding dimension from the aggregator
        hidden_dims: List of hidden layer dimensions for the MLP
        num_betas: Number of shape parameters to predict (default: 10)
        dropout: Dropout rate
        activation: Activation function type ('gelu', 'relu', 'leaky_relu')
        use_smpl_mean: Whether to use mean SMPL parameters as initialization
    """
    
    # SMPL parameter dimensions
    NUM_GLOBAL_ORIENT = 3   # Root rotation (axis-angle)
    NUM_BODY_POSE = 69      # 23 joints × 3 (axis-angle)
    NUM_TRANSL = 3          # Translation
    
    def __init__(
        self,
        losses,
        emb_size: int = 512,
        hidden_dims: list = [1024, 512],
        num_betas: int = 10,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_smpl_mean: bool = True,
    ):
        super().__init__(losses)
        
        self.emb_size = emb_size
        self.num_betas = num_betas
        self.use_smpl_mean = use_smpl_mean
        
        # Total output dimensions
        self.num_output = (
            self.NUM_GLOBAL_ORIENT + 
            self.NUM_BODY_POSE + 
            num_betas + 
            self.NUM_TRANSL
        )  # 3 + 69 + 10 + 3 = 85
        
        # Build MLP layers
        layers = []
        in_dim = emb_size
        
        # Get activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.GELU
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(in_dim, self.num_output))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize mean SMPL parameters (used as initialization bias)
        if use_smpl_mean:
            self._init_smpl_mean()
    
    def _init_smpl_mean(self):
        """Initialize mean SMPL parameters for better training stability."""
        # Mean pose is approximately zeros (T-pose)
        mean_global_orient = torch.zeros(self.NUM_GLOBAL_ORIENT)
        mean_body_pose = torch.zeros(self.NUM_BODY_POSE)
        mean_betas = torch.zeros(self.num_betas)
        mean_transl = torch.zeros(self.NUM_TRANSL)
        
        # Concatenate all mean parameters
        mean_params = torch.cat([
            mean_global_orient,
            mean_body_pose,
            mean_betas,
            mean_transl
        ])
        
        self.register_buffer('mean_params', mean_params)
    
    def _extract_skeleton_token(self, x):
        """Extract skeleton/joint token from aggregated features.
        
        Args:
            x: Aggregated features, can be:
               - List of features from different layers
               - Tensor of shape (B, M, T, N, C)
        
        Returns:
            Tensor of shape (B, C)
        """
        if isinstance(x, list):
            x = x[-1]  # Use last layer features
        
        # x.shape: B, M, T, N, C
        # N contains: [camera_token, joint_token, register_tokens..., patch_tokens...]
        B, M, T, N, C = x.shape
        
        # Use the skeleton token (index 1) - same convention as RegressionKeypointHead
        # Or use the last token which is typically the skeleton token
        x = x[:, :, :, -1, :]  # (B, M, T, C)
        
        # Average over modalities and temporal dimension
        x = x.mean(dim=[1, 2])  # (B, C)
        
        return x
    
    def forward(self, x):
        """Forward pass to predict SMPL parameters.
        
        Args:
            x: Aggregated features from the model
        
        Returns:
            Dictionary with predicted SMPL parameters:
                - global_orient: (B, 3)
                - body_pose: (B, 69)
                - betas: (B, 10)
                - transl: (B, 3)
        """
        # Extract features
        feat = self._extract_skeleton_token(x)  # (B, C)
        
        # Predict parameters through MLP
        params = self.mlp(feat)  # (B, 85)
        
        # Add mean parameters if using
        if self.use_smpl_mean:
            params = params + self.mean_params
        
        # Split into individual components
        idx = 0
        global_orient = params[:, idx:idx + self.NUM_GLOBAL_ORIENT]
        idx += self.NUM_GLOBAL_ORIENT
        
        body_pose = params[:, idx:idx + self.NUM_BODY_POSE]
        idx += self.NUM_BODY_POSE
        
        betas = params[:, idx:idx + self.num_betas]
        idx += self.num_betas
        
        transl = params[:, idx:idx + self.NUM_TRANSL]
        
        return {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'betas': betas,
            'transl': transl,
        }
    
    def loss(self, x, data_batch):
        """Compute loss for SMPL parameter regression.
        
        Args:
            x: Aggregated features
            data_batch: Dictionary containing ground truth data including 'gt_smpl'
        
        Returns:
            Dictionary of losses
        """
        pred_smpl = self.forward(x)
        
        # Get ground truth SMPL parameters
        gt_smpl = data_batch['gt_smpl']
        
        # Handle different gt_smpl formats (list of dicts or dict of tensors)
        if isinstance(gt_smpl, list):
            # Convert list of dicts to dict of tensors
            gt_global_orient = torch.stack([
                s['global_orient'] if isinstance(s['global_orient'], torch.Tensor) 
                else torch.from_numpy(s['global_orient']) 
                for s in gt_smpl
            ]).to(pred_smpl['global_orient'].device)
            gt_body_pose = torch.stack([
                s['body_pose'] if isinstance(s['body_pose'], torch.Tensor)
                else torch.from_numpy(s['body_pose'])
                for s in gt_smpl
            ]).to(pred_smpl['body_pose'].device)
            gt_betas = torch.stack([
                s['betas'] if isinstance(s['betas'], torch.Tensor)
                else torch.from_numpy(s['betas'])
                for s in gt_smpl
            ]).to(pred_smpl['betas'].device)
            gt_transl = torch.stack([
                s['transl'] if isinstance(s['transl'], torch.Tensor)
                else torch.from_numpy(s['transl'])
                for s in gt_smpl
            ]).to(pred_smpl['transl'].device)
        else:
            gt_global_orient = gt_smpl['global_orient'].to(pred_smpl['global_orient'].device)
            gt_body_pose = gt_smpl['body_pose'].to(pred_smpl['body_pose'].device)
            gt_betas = gt_smpl['betas'].to(pred_smpl['betas'].device)
            gt_transl = gt_smpl['transl'].to(pred_smpl['transl'].device)
        
        # Ensure float type
        gt_global_orient = gt_global_orient.float()
        gt_body_pose = gt_body_pose.float()
        gt_betas = gt_betas.float()
        gt_transl = gt_transl.float()
        
        # Concatenate all predictions and ground truths for loss computation
        pred_all = torch.cat([
            pred_smpl['global_orient'],
            pred_smpl['body_pose'],
            pred_smpl['betas'],
            pred_smpl['transl']
        ], dim=1)
        
        gt_all = torch.cat([
            gt_global_orient,
            gt_body_pose,
            gt_betas[:, :self.num_betas],  # Only use first num_betas
            gt_transl
        ], dim=1)
        
        # Compute losses
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[loss_name] = (loss_fn(pred_all, gt_all), loss_weight)
        
        return losses
    
    def predict(self, x):
        """Predict SMPL parameters.
        
        Returns only SMPL parameters. Conversion to keypoints/vertices
        is handled by metrics and visualization code.
        
        Args:
            x: Aggregated features
        
        Returns:
            Dictionary with SMPL parameters:
                - global_orient: (B, 3)
                - body_pose: (B, 69)
                - betas: (B, 10)
                - transl: (B, 3)
        """
        return self.forward(x)


class SMPLHeadSimple(BaseHead):
    """
    Simplified SMPL head that only predicts pose parameters (no shape).
    
    Useful when shape is fixed or when training with limited data.
    """
    
    NUM_GLOBAL_ORIENT = 3
    NUM_BODY_POSE = 69
    NUM_TRANSL = 3
    
    def __init__(
        self,
        losses,
        emb_size: int = 512,
        hidden_dims: list = [512],
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__(losses)
        
        self.emb_size = emb_size
        self.num_output = self.NUM_GLOBAL_ORIENT + self.NUM_BODY_POSE + self.NUM_TRANSL
        
        # Build MLP
        layers = []
        in_dim = emb_size
        
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            act_fn = nn.GELU
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, self.num_output))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize mean parameters
        mean_params = torch.zeros(self.num_output)
        self.register_buffer('mean_params', mean_params)
    
    def _extract_skeleton_token(self, x):
        if isinstance(x, list):
            x = x[-1]
        B, M, T, N, C = x.shape
        x = x[:, :, :, -1, :]
        x = x.mean(dim=[1, 2])
        return x
    
    def forward(self, x):
        feat = self._extract_skeleton_token(x)
        params = self.mlp(feat) + self.mean_params
        
        idx = 0
        global_orient = params[:, idx:idx + self.NUM_GLOBAL_ORIENT]
        idx += self.NUM_GLOBAL_ORIENT
        body_pose = params[:, idx:idx + self.NUM_BODY_POSE]
        idx += self.NUM_BODY_POSE
        transl = params[:, idx:idx + self.NUM_TRANSL]
        
        return {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'transl': transl,
        }
    
    def loss(self, x, data_batch):
        pred = self.forward(x)
        gt_smpl = data_batch['gt_smpl']
        
        if isinstance(gt_smpl, list):
            gt_global_orient = torch.stack([
                torch.from_numpy(s['global_orient']) if isinstance(s['global_orient'], np.ndarray)
                else s['global_orient'] for s in gt_smpl
            ]).to(pred['global_orient'].device).float()
            gt_body_pose = torch.stack([
                torch.from_numpy(s['body_pose']) if isinstance(s['body_pose'], np.ndarray)
                else s['body_pose'] for s in gt_smpl
            ]).to(pred['body_pose'].device).float()
            gt_transl = torch.stack([
                torch.from_numpy(s['transl']) if isinstance(s['transl'], np.ndarray)
                else s['transl'] for s in gt_smpl
            ]).to(pred['transl'].device).float()
        else:
            gt_global_orient = gt_smpl['global_orient'].to(pred['global_orient'].device).float()
            gt_body_pose = gt_smpl['body_pose'].to(pred['body_pose'].device).float()
            gt_transl = gt_smpl['transl'].to(pred['transl'].device).float()
        
        pred_all = torch.cat([pred['global_orient'], pred['body_pose'], pred['transl']], dim=1)
        gt_all = torch.cat([gt_global_orient, gt_body_pose, gt_transl], dim=1)
        
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[loss_name] = (loss_fn(pred_all, gt_all), loss_weight)
        
        return losses
    
    def predict(self, x):
        """Returns predicted pose parameters as flattened keypoint-like output."""
        pred = self.forward(x)
        # Return concatenated parameters reshaped as pseudo-keypoints for metric compatibility
        B = pred['global_orient'].shape[0]
        pose_flat = torch.cat([
            pred['global_orient'],
            pred['body_pose'],
            pred['transl']
        ], dim=1)
        # Reshape to (B, 25, 3) - close to 24 joints + 1 extra
        return pose_flat.view(B, -1, 3)[:, :24, :]  # (B, 24, 3)
