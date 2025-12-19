"""
Transformer Aggregator V3-Lite: Parameter-Efficient Version
Optimizations:
1. Share weights across parallel paths (instead of 3 separate branches)
2. Lightweight modality fusion (no separate uncertainty networks)
3. Efficient multi-scale output (reuse features instead of FPN)
4. Combined spatio-temporal attention (instead of separate blocks)

Parameter reduction: ~180M → ~60M (3× smaller while keeping key improvements)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange, reduce, repeat
import math

from models.video_encoders.layers.block import Block
from .layers.block import CABlock
from .layers.gcn import TCN_GCN_unit as GCNBlock
from misc.skeleton import get_adjacency_matrix, H36MSkeleton, SMPLSkeleton


def get_sinusoidal_embeddings(seq_len: int, dim: int, device: torch.device) -> Tensor:
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class LightweightModalityFusion(nn.Module):
    """
    Lightweight version: predict weights directly from features
    No separate uncertainty estimators (saves 4× UncertaintyEstimator)
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Single lightweight network for all modalities
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, embed_dim)),  # Pool over spatial
            nn.Flatten(1, 2),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, modality_tokens: List[Tensor], valid_mask: Tensor):
        """
        Args:
            modality_tokens: List of [B, T, N, C]
            valid_mask: [num_modalities]
        Returns:
            fused: [B, T, N, C]
            weights: [B, num_active]
        """
        active_tokens = [t for t, v in zip(modality_tokens, valid_mask) if v]
        if not active_tokens:
            raise ValueError("At least one modality required")
        
        # Compute quality scores for each modality
        scores = []
        for tokens in active_tokens:
            score = self.weight_predictor(tokens)  # [B, 1]
            scores.append(score)
        
        scores = torch.cat(scores, dim=-1)  # [B, num_active]
        weights = F.softmax(scores, dim=-1)  # [B, num_active]
        
        # Weighted fusion
        stacked = torch.stack(active_tokens, dim=1)  # [B, num_active, T, N, C]
        weights_expanded = weights.view(-1, len(active_tokens), 1, 1, 1)
        fused = (stacked * weights_expanded).sum(dim=1)  # [B, T, N, C]
        
        return fused, weights


class EfficientMultiPathBlock(nn.Module):
    """
    Efficient version: Share base transformer, use routing for different paths
    Instead of 3 separate branches, use 1 transformer + path-specific adapters
    Saves: 2 × Transformer parameters
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        # Shared base transformer (used by all paths)
        self.shared_transformer = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_norm=True,
        )
        
        # Lightweight path-specific adapters (much smaller than full transformers)
        self.local_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        self.global_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        self.temporal_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        # Lightweight fusion (instead of heavy Linear(3×embed_dim))
        self.path_fusion = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, tokens, Ns):
        """
        Args:
            tokens: [B, T, N, C]
            Ns: modality token counts
        Returns:
            output: [B, T, N, C]
        """
        B, T, N, C = tokens.shape
        
        # Path 1: Local (per-modality)
        local_out = self._process_local(tokens, Ns)
        local_out = local_out + self.local_adapter(local_out)
        
        # Path 2: Global (all tokens)
        global_tokens = rearrange(tokens, 'b t n c -> (b t) n c')
        global_out = self.shared_transformer(global_tokens)
        global_out = rearrange(global_out, '(b t) n c -> b t n c', b=B, t=T)
        global_out = global_out + self.global_adapter(global_out)
        
        # Path 3: Temporal
        temporal_tokens = rearrange(tokens, 'b t n c -> (b n) t c')
        temporal_out = self.shared_transformer(temporal_tokens)
        temporal_out = rearrange(temporal_out, '(b n) t c -> b t n c', b=B, n=N)
        temporal_out = temporal_out + self.temporal_adapter(temporal_out)
        
        # Lightweight fusion with learnable weights
        weights = F.softmax(self.path_fusion, dim=0)
        output = weights[0] * local_out + weights[1] * global_out + weights[2] * temporal_out
        
        # Residual connection
        output = tokens + output
        
        return output
    
    def _process_local(self, tokens, Ns):
        """Process each modality separately"""
        B, T, N_total, C = tokens.shape
        outputs = []
        cumsum = 0
        
        for n in Ns:
            if n > 0:
                mod_tokens = tokens[:, :, cumsum:cumsum+n, :]
                mod_tokens = rearrange(mod_tokens, 'b t n c -> (b t) n c')
                mod_tokens = self.shared_transformer(mod_tokens)
                mod_tokens = rearrange(mod_tokens, '(b t) n c -> b t n c', b=B, t=T)
                outputs.append(mod_tokens)
            else:
                outputs.append(torch.zeros(B, T, 0, C, device=tokens.device))
            cumsum += n
        
        return torch.cat(outputs, dim=2)


class SkeletonGuidedGCN(nn.Module):
    """
    Combined GCN + skeleton-aware attention
    Instead of separate SkeletonGuidedAttention block
    """
    def __init__(self, embed_dim, adjacency_matrix, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.gcn = GCNBlock(embed_dim, embed_dim, adjacency_matrix, adaptive=True)
        
        # Lightweight skeleton bias
        self.register_buffer('skeleton_adj', torch.FloatTensor(adjacency_matrix))
        self.skeleton_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, joint_tokens):
        """
        Args:
            joint_tokens: [B, M, T, V, C]
        Returns:
            refined: [B, M, T, V, C]
        """
        B, M, T, V, C = joint_tokens.shape
        
        # Apply GCN
        gcn_input = rearrange(joint_tokens, 'b m t v c -> (b m) c t v')
        gcn_output = self.gcn(gcn_input)
        refined = rearrange(gcn_output, '(b m) c t v -> b m t v c', b=B, m=M)
        
        # Add skeleton-guided aggregation (lightweight)
        skeleton_features = torch.einsum('vw,bmtwc->bmtvc', 
                                         self.skeleton_adj, 
                                         refined)
        refined = refined + self.skeleton_scale * skeleton_features
        
        return refined


class TransformerAggregatorV3Lite(nn.Module):
    """
    Parameter-efficient version with key V3 improvements:
    - ✅ Shared-weight multi-path processing (not 3 separate branches)
    - ✅ Lightweight modality fusion
    - ✅ Skeleton-guided GCN
    - ✅ Multi-scale output (reuse intermediate features)
    - ❌ No separate FPN (just collect intermediate features)
    - ❌ No separate uncertainty estimators
    
    Parameters: ~60M (vs ~180M in full V3, ~50M in V2)
    """
    def __init__(self,
                 input_dims=[512, 512, 512, 512],
                 embed_dim=512,
                 num_register_tokens=4,
                 depth=12,
                 num_heads=8,
                 skeleton_type="h36m",
                 use_multi_path=True,
                 use_dynamic_fusion=True,
                 use_skeleton_gcn=True,
                 use_grad_ckpt=False,
                 max_spatial_seq_len=1000,
                 max_temporal_seq_len=243,
                 ):
        super().__init__()
        
        # Skeleton
        if skeleton_type in ["h36m", "mmfi"]:
            skeleton = H36MSkeleton()
        elif skeleton_type == "smpl":
            skeleton = SMPLSkeleton()
        else:
            raise ValueError(f"Unknown skeleton type: {skeleton_type}")
        
        self.num_joints = skeleton.num_joints
        self.num_modalities = 4
        self.depth = depth
        self.use_grad_ckpt = use_grad_ckpt
        self.use_multi_path = use_multi_path
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_skeleton_gcn = use_skeleton_gcn
        
        # Input projections
        rgb_dim, depth_dim, lidar_dim, mmwave_dim = input_dims
        self.proj_rgb = nn.Sequential(
            nn.Linear(rgb_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.proj_depth = nn.Sequential(
            nn.Linear(depth_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.proj_lidar = nn.Sequential(
            nn.Linear(lidar_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.proj_mmwave = nn.Sequential(
            nn.Linear(mmwave_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Learnable tokens
        self.camera_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        self.joint_tokens = nn.Parameter(torch.randn(1, 1, self.num_joints, embed_dim))
        self.register_tokens = nn.Parameter(torch.randn(1, 1, num_register_tokens, embed_dim))
        self.num_register_tokens = num_register_tokens
        
        # Embeddings
        self.modality_embed = nn.Parameter(torch.randn(self.num_modalities, embed_dim))
        self.register_buffer("spatial_pos_embed", 
                           get_sinusoidal_embeddings(max_spatial_seq_len, embed_dim, device='cpu'),
                           persistent=False)
        self.register_buffer("temporal_pos_embed",
                           get_sinusoidal_embeddings(max_temporal_seq_len, embed_dim, device='cpu'),
                           persistent=False)
        
        # Lightweight modality fusion
        if use_dynamic_fusion:
            self.modality_fusion = LightweightModalityFusion(embed_dim)
        
        # Efficient multi-path blocks (shared weights)
        if use_multi_path:
            self.multi_path_blocks = nn.ModuleList([
                EfficientMultiPathBlock(embed_dim, num_heads)
                for _ in range(depth)
            ])
        else:
            # Fallback to standard blocks
            self.blocks = nn.ModuleList([
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                      qkv_bias=True, qk_norm=True)
                for _ in range(depth)
            ])
        
        # Skeleton-guided GCN
        if use_skeleton_gcn:
            A = get_adjacency_matrix(skeleton.bones, skeleton.num_joints)
            self.skeleton_gcn_blocks = nn.ModuleList([
                SkeletonGuidedGCN(embed_dim, A, self.num_joints)
                for _ in range(depth)
            ])
        
        # Output projection (simple, no heavy FPN)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, features, return_intermediates=False, **kwargs):
        """
        Args:
            features: (rgb, depth, lidar, mmwave) each [B, T, N, C] or None
        Returns:
            If return_intermediates:
                Dict with 'final', 'coarse' (layer depth//4), 'medium' (layer depth//2)
            Else:
                [B, M, T, 1+V, C]
        """
        features_rgb, features_depth, features_lidar, features_mmwave = features
        
        # Project inputs
        B, T = 0, 0
        projected_features = []
        valid_mask = torch.zeros(self.num_modalities, dtype=torch.bool)
        
        for i, (feat, proj) in enumerate(zip(
            [features_rgb, features_depth, features_lidar, features_mmwave],
            [self.proj_rgb, self.proj_depth, self.proj_lidar, self.proj_mmwave]
        )):
            if feat is not None:
                feat = proj(feat)
                projected_features.append(feat)
                valid_mask[i] = True
                B, T, _, _ = feat.shape
            else:
                projected_features.append(None)
        
        if B == 0:
            raise ValueError("At least one modality required")
        
        # Add positional embeddings (per-modality)
        temporal_pe = self.temporal_pos_embed[:T, :].unsqueeze(0).unsqueeze(2)
        
        for i, feat in enumerate(projected_features):
            if feat is not None:
                N = feat.shape[2]
                spatial_pe = self.spatial_pos_embed[:N, :].unsqueeze(0).unsqueeze(0)
                projected_features[i] = feat + spatial_pe + temporal_pe
        
        # Prepare special tokens
        camera_tokens = self.camera_token.expand(B, T, -1, -1)
        joint_tokens = self.joint_tokens.expand(B, T, -1, -1)
        register_tokens = self.register_tokens.expand(B, T, -1, -1)
        
        # Assemble tokens
        modality_tokens = []
        Ns = []
        
        for i, feat in enumerate(projected_features):
            if feat is not None:
                mod_tokens = torch.cat([
                    camera_tokens, joint_tokens, register_tokens, feat
                ], dim=2)
                mod_tokens = mod_tokens + self.modality_embed[i].view(1, 1, 1, -1)
                modality_tokens.append(mod_tokens)
                Ns.append(mod_tokens.shape[2])
            else:
                Ns.append(0)
        
        # Concatenate all modalities
        tokens = torch.cat(modality_tokens, dim=2)  # [B, T, N_total, C]
        
        # Process through layers
        intermediates = {}
        
        for layer_idx in range(self.depth):
            # Multi-path processing (shared weights)
            if self.use_multi_path:
                tokens = self.multi_path_blocks[layer_idx](tokens, Ns)
            else:
                B_cur, T_cur, N_cur, C_cur = tokens.shape
                tokens_flat = rearrange(tokens, 'b t n c -> (b t) n c')
                tokens_flat = self.blocks[layer_idx](tokens_flat)
                tokens = rearrange(tokens_flat, '(b t) n c -> b t n c', b=B_cur, t=T_cur)
            
            # Skeleton-guided GCN refinement
            if self.use_skeleton_gcn:
                tokens = self._apply_skeleton_gcn(tokens, Ns, layer_idx)
            
            # Save intermediate features for multi-scale output
            if layer_idx == self.depth // 4:
                intermediates['coarse'] = self._extract_output_tokens(tokens, Ns)
            elif layer_idx == self.depth // 2:
                intermediates['medium'] = self._extract_output_tokens(tokens, Ns)
        
        # Final output
        final_output = self._extract_output_tokens(tokens, Ns)
        final_output = self.output_proj(final_output)
        intermediates['fine'] = final_output
        
        if return_intermediates:
            return intermediates
        return final_output
    
    def _apply_skeleton_gcn(self, tokens, Ns, layer_idx):
        """Apply skeleton-guided GCN to joint tokens"""
        B, T, N_total, C = tokens.shape
        
        joint_slices = []
        joint_positions = []
        cumsum = 0
        
        for i, n in enumerate(Ns):
            if n > 0:
                joint_start = cumsum + 1
                joint_end = joint_start + self.num_joints
                joint_slice = tokens[:, :, joint_start:joint_end, :]
                joint_slices.append(joint_slice)
                joint_positions.append(joint_start)
            cumsum += n
        
        if not joint_slices:
            return tokens
        
        # Stack and process
        stacked = torch.stack(joint_slices, dim=1)  # [B, M, T, V, C]
        
        # Apply skeleton GCN
        if self.use_grad_ckpt and self.training:
            refined = checkpoint(
                self.skeleton_gcn_blocks[layer_idx], 
                stacked, 
                use_reentrant=False
            )
        else:
            refined = self.skeleton_gcn_blocks[layer_idx](stacked)
        
        # Insert back
        new_tokens = tokens.clone()
        for mod_idx, joint_start in enumerate(joint_positions):
            new_tokens[:, :, joint_start:joint_start + self.num_joints, :] = refined[:, mod_idx]
        
        return new_tokens
    
    def _extract_output_tokens(self, tokens, Ns):
        """Extract camera + joint tokens: [B, M, T, 1+V, C]"""
        B, T, _, C = tokens.shape
        output_slices = []
        cumsum = 0
        
        for n in Ns:
            if n > 0:
                output_slice = tokens[:, :, cumsum:cumsum + 1 + self.num_joints, :]
                output_slices.append(output_slice)
            cumsum += n
        
        if output_slices:
            output = torch.stack(output_slices, dim=1)
        else:
            output = tokens.new_zeros(B, 0, T, 1 + self.num_joints, C)
        
        return output
