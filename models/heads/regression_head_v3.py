"""
Regression Head V3 for TransformerAggregatorV3
Features:
1. Multi-scale feature fusion (coarse, medium, fine)
2. Uncertainty estimation
3. Modality-wise attention weights
4. Temporal aggregation strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional

from .base_head import BaseHead


class ModalityAttentionPooling(nn.Module):
    """
    Learn to weight different modalities based on their reliability
    """
    def __init__(self, embed_dim, num_modalities=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, M, T, J, C] features from different modalities
        Returns:
            pooled: [B, T, J, C] - modality-weighted features
            weights: [B, M, T, J] - attention weights
        """
        B, M, T, J, C = x.shape
        
        # Compute attention scores for each modality
        x_flat = rearrange(x, 'b m t j c -> (b m t j) c')
        attn_scores = self.attention(x_flat)  # [B*M*T*J, 1]
        attn_scores = rearrange(attn_scores, '(b m t j) 1 -> b m t j', b=B, m=M, t=T, j=J)
        
        # Softmax over modality dimension
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, M, T, J]
        
        # Weighted sum
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [B, M, T, J, 1]
        pooled = (x * attn_weights_expanded).sum(dim=1)  # [B, T, J, C]
        
        return pooled, attn_weights


class TemporalAggregation(nn.Module):
    """
    Aggregate temporal information with multiple strategies
    """
    def __init__(self, embed_dim, strategy='attention'):
        super().__init__()
        self.strategy = strategy
        
        if strategy == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1)
            )
        elif strategy == 'conv':
            self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, J, C]
        Returns:
            aggregated: [B, J, C]
            weights: [B, T, J] if using attention, else None
        """
        B, T, J, C = x.shape
        
        if self.strategy == 'mean':
            return x.mean(dim=1), None
        
        elif self.strategy == 'max':
            return x.max(dim=1)[0], None
        
        elif self.strategy == 'last':
            return x[:, -1], None
        
        elif self.strategy == 'attention':
            # Compute temporal attention weights
            x_flat = rearrange(x, 'b t j c -> (b t j) c')
            attn_scores = self.temporal_attention(x_flat)  # [B*T*J, 1]
            attn_scores = rearrange(attn_scores, '(b t j) 1 -> b t j', b=B, t=T, j=J)
            
            # Softmax over time
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, T, J]
            
            # Weighted sum
            attn_weights_expanded = attn_weights.unsqueeze(-1)  # [B, T, J, 1]
            aggregated = (x * attn_weights_expanded).sum(dim=1)  # [B, J, C]
            
            return aggregated, attn_weights
        
        elif self.strategy == 'conv':
            # Apply 1D convolution over time
            x_permuted = rearrange(x, 'b t j c -> (b j) c t')
            x_conv = self.temporal_conv(x_permuted)
            x_conv = rearrange(x_conv, '(b j) c t -> b t j c', b=B, j=J)
            # Take the middle frame
            return x_conv[:, T//2], None
        
        else:
            raise ValueError(f"Unknown temporal aggregation strategy: {self.strategy}")


class MultiScaleFusion(nn.Module):
    """
    Fuse multi-scale features (coarse, medium, fine)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # Learnable weights for 3 scales
        
        # Optional: add learnable projections
        self.proj_coarse = nn.Linear(embed_dim, embed_dim)
        self.proj_medium = nn.Linear(embed_dim, embed_dim)
        self.proj_fine = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, coarse, medium, fine):
        """
        Args:
            coarse, medium, fine: [B, M, T, J, C] or None
        Returns:
            fused: [B, M, T, J, C]
        """
        features = []
        weights = []
        
        if coarse is not None:
            features.append(self.proj_coarse(coarse))
            weights.append(self.fusion_weights[0])
        
        if medium is not None:
            features.append(self.proj_medium(medium))
            weights.append(self.fusion_weights[1])
        
        if fine is not None:
            features.append(self.proj_fine(fine))
            weights.append(self.fusion_weights[2])
        
        if not features:
            raise ValueError("At least one scale must be provided")
        
        # Normalize weights
        weights = torch.stack(weights)
        weights = F.softmax(weights, dim=0)
        
        # Weighted sum
        fused = sum(w * f for w, f in zip(weights, features))
        
        return fused


class UncertaintyEstimationHead(nn.Module):
    """
    Estimate per-joint uncertainty (aleatoric uncertainty)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 3),  # 3D uncertainty (sigma_x, sigma_y, sigma_z)
            nn.Softplus()  # Ensure positive
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, J, C]
        Returns:
            uncertainty: [B, J, 3] - per-joint 3D uncertainty
        """
        return self.uncertainty_mlp(x)


class RegressionKeypointHeadV3(BaseHead):
    """
    Advanced regression head for V3 aggregator with:
    - Multi-scale feature fusion
    - Modality-aware attention
    - Temporal aggregation
    - Uncertainty estimation
    """
    def __init__(self, 
                 losses, 
                 emb_size=512,
                 use_multi_scale=True,
                 use_modality_attention=True,
                 temporal_strategy='attention',  # 'mean', 'max', 'last', 'attention', 'conv'
                 estimate_uncertainty=False,
                 dropout=0.1):
        super().__init__(losses)
        
        self.emb_size = emb_size
        self.use_multi_scale = use_multi_scale
        self.use_modality_attention = use_modality_attention
        self.estimate_uncertainty = estimate_uncertainty
        
        # Multi-scale fusion
        if use_multi_scale:
            self.multi_scale_fusion = MultiScaleFusion(emb_size)
        
        # Modality attention pooling
        if use_modality_attention:
            self.modality_pooling = ModalityAttentionPooling(emb_size)
        
        # Temporal aggregation
        self.temporal_aggregation = TemporalAggregation(emb_size, strategy=temporal_strategy)
        
        # Main regression head
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        
        self.regression_mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size // 2, 3)  # 3D keypoint coordinates
        )
        
        # Uncertainty estimation head
        if estimate_uncertainty:
            self.uncertainty_head = UncertaintyEstimationHead(emb_size)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Either dict (multi-scale) or tensor (single-scale)
               Dict: {'coarse': [B, M, T, 1+J, C], 'medium': ..., 'fine': ...}
               Tensor: [B, M, T, 1+J, C]
        Returns:
            pred_keypoints: [B, J, 3]
            If estimate_uncertainty: also returns uncertainty [B, J, 3]
            If return_attention: also returns attention weights
        """
        outputs = {}
        
        # Handle multi-scale inputs
        if isinstance(x, dict):
            # Extract features from multi-scale outputs
            coarse = x.get('coarse', None)
            medium = x.get('medium', None)
            fine = x.get('fine', None)
            
            # Remove camera token (first token), keep only joint tokens
            if coarse is not None:
                coarse = coarse[:, :, :, 1:, :]  # [B, M, T, J, C]
            if medium is not None:
                medium = medium[:, :, :, 1:, :]
            if fine is not None:
                fine = fine[:, :, :, 1:, :]
            
            # Fuse multi-scale features
            if self.use_multi_scale:
                features = self.multi_scale_fusion(coarse, medium, fine)
            else:
                # Use only fine features if available, else fall back
                features = fine if fine is not None else (medium if medium is not None else coarse)
        
        elif isinstance(x, list):
            # Handle list of intermediate outputs (use last one)
            x = x[-1]
            features = x[:, :, :, 1:, :]  # [B, M, T, J, C]
        
        else:
            # Single tensor input
            features = x[:, :, :, 1:, :]  # [B, M, T, J, C]
        
        B, M, T, J, C = features.shape
        
        # Modality attention pooling
        if self.use_modality_attention:
            features, modality_weights = self.modality_pooling(features)  # [B, T, J, C]
            outputs['modality_weights'] = modality_weights
        else:
            # Simple average over modalities
            features = features.mean(dim=1)  # [B, T, J, C]
            modality_weights = None
        
        # Temporal aggregation
        features, temporal_weights = self.temporal_aggregation(features)  # [B, J, C]
        if temporal_weights is not None:
            outputs['temporal_weights'] = temporal_weights
        
        # Normalization and dropout
        features = self.norm(features)
        features = self.dropout(features)
        
        # Regression to 3D coordinates
        pred_keypoints = self.regression_mlp(features)  # [B, J, 3]
        
        # Uncertainty estimation
        if self.estimate_uncertainty:
            uncertainty = self.uncertainty_head(features)  # [B, J, 3]
            outputs['uncertainty'] = uncertainty
        
        outputs['keypoints'] = pred_keypoints
        
        if return_attention:
            return outputs
        else:
            if self.estimate_uncertainty:
                return pred_keypoints, outputs['uncertainty']
            else:
                return pred_keypoints
    
    def loss(self, x, data_batch):
        """
        Compute losses
        """
        # Forward pass with attention weights for monitoring
        outputs = self.forward(x, return_attention=True)
        pred_keypoints = outputs['keypoints']
        
        losses = {}
        
        # Main regression losses
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            if self.estimate_uncertainty and hasattr(loss_fn, 'use_uncertainty'):
                # Use uncertainty-aware loss if available
                loss_val = loss_fn(
                    pred_keypoints, 
                    data_batch['gt_keypoints'],
                    uncertainty=outputs.get('uncertainty', None)
                )
            else:
                loss_val = loss_fn(pred_keypoints, data_batch['gt_keypoints'])
            
            losses[loss_name] = (loss_val, loss_weight)
        
        # Add attention weights for monitoring (zero weight)
        if 'modality_weights' in outputs:
            losses['modality_attention'] = (outputs['modality_weights'].mean(), 0.0)
        if 'temporal_weights' in outputs:
            losses['temporal_attention'] = (outputs['temporal_weights'].mean(), 0.0)
        if 'uncertainty' in outputs:
            losses['uncertainty_mean'] = (outputs['uncertainty'].mean(), 0.0)
        
        return losses
    
    def predict(self, x):
        """
        Inference mode
        """
        with torch.no_grad():
            result = self.forward(x, return_attention=False)
            
            if self.estimate_uncertainty:
                pred_keypoints, uncertainty = result
                return {
                    'keypoints': pred_keypoints,
                    'uncertainty': uncertainty
                }
            else:
                return {'keypoints': result}


class RegressionKeypointHeadV3Simple(BaseHead):
    """
    Simplified version for compatibility with non-dict outputs
    Similar to V2 but with better feature processing
    """
    def __init__(self, losses, emb_size=512, dropout=0.1):
        super().__init__(losses)
        self.emb_size = emb_size
        
        # Modality attention
        self.modality_attention = nn.Sequential(
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, 1)
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, 1)
        )
        
        # Regression head
        self.norm = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size // 2, 3)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, M, T, 1+J, C] or list of such tensors or dict
        Returns:
            pred_keypoints: [B, J, 3]
        """
        # Handle different input formats
        if isinstance(x, dict):
            # Use fine features if available
            x = x.get('fine', x.get('medium', x.get('coarse')))
        elif isinstance(x, list):
            x = x[-1]
        
        # x: [B, M, T, 1+J, C]
        B, M, T, N, C = x.shape
        x = x[:, :, :, 1:, :]  # Remove camera token -> [B, M, T, J, C]
        J = N - 1
        
        # Modality attention
        x_for_mod_attn = rearrange(x, 'b m t j c -> (b t j) m c')
        mod_attn_scores = self.modality_attention(x_for_mod_attn)  # [B*T*J, M, 1]
        mod_attn_weights = F.softmax(mod_attn_scores, dim=1)
        x_for_mod_attn = x_for_mod_attn * mod_attn_weights
        x = rearrange(x_for_mod_attn, '(b t j) m c -> b m t j c', b=B, t=T, j=J)
        x = x.sum(dim=1)  # [B, T, J, C]
        
        # Temporal attention
        x_for_temp_attn = rearrange(x, 'b t j c -> (b j) t c')
        temp_attn_scores = self.temporal_attention(x_for_temp_attn)  # [B*J, T, 1]
        temp_attn_weights = F.softmax(temp_attn_scores, dim=1)
        x_for_temp_attn = x_for_temp_attn * temp_attn_weights
        x = rearrange(x_for_temp_attn, '(b j) t c -> b t j c', b=B, j=J)
        x = x.sum(dim=1)  # [B, J, C]
        
        # Regression
        x = self.norm(x)
        x = self.mlp(x)
        
        return x
    
    def loss(self, x, data_batch):
        pred_keypoints = self.forward(x)
        
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[loss_name] = (loss_fn(pred_keypoints, data_batch['gt_keypoints']), loss_weight)
        
        return losses
    
    def predict(self, x):
        return self.forward(x)
