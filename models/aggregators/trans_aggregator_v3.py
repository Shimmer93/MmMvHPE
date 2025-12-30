"""
Improved Transformer Aggregator V3
Key improvements:
1. Parallel multi-path architecture (local, global, temporal branches)
2. Dynamic modality fusion with uncertainty modeling
3. Hierarchical spatio-temporal modeling
4. Skeleton-guided attention
5. Better output fusion strategy (FPN-style)
6. Modular and extensible design
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


class ModalityUncertaintyEstimator(nn.Module):
    """Estimate uncertainty/confidence for each modality"""
    def __init__(self, embed_dim):
        super().__init__()
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, tokens):
        """
        Args:
            tokens: [B, T, N, C]
        Returns:
            uncertainty: [B, T, 1] - scalar uncertainty per sample per frame
        """
        # Global average pooling over spatial dimension
        feat = tokens.mean(dim=2)  # [B, T, C]
        uncertainty = self.uncertainty_head(feat)  # [B, T, 1]
        return uncertainty


class DynamicModalityFusion(nn.Module):
    """
    Dynamic fusion module that adaptively weighs different modalities
    based on their estimated reliability and inter-modality relationships
    """
    def __init__(self, embed_dim, num_modalities=4):
        super().__init__()
        self.num_modalities = num_modalities
        
        # Uncertainty estimators for each modality path
        self.uncertainty_estimators = nn.ModuleList([
            ModalityUncertaintyEstimator(embed_dim) 
            for _ in range(num_modalities)
        ])
        
        # Modality affinity network (learns inter-modality relationships)
        self.affinity_net = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_modalities * num_modalities),
        )
        
        # Fusion weights predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(embed_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, modality_tokens: List[Tensor], valid_mask: Tensor):
        """
        Args:
            modality_tokens: List of [B, T, N, C] tensors (one per modality)
            valid_mask: [num_modalities] boolean mask indicating which modalities are present
        Returns:
            fused: [B, T, N, C] fused features
            weights: [B, T, num_modalities] fusion weights
            uncertainties: [B, T, num_modalities] uncertainty estimates
        """
        active_tokens = [t for t, v in zip(modality_tokens, valid_mask) if v]
        if not active_tokens:
            raise ValueError("At least one modality must be present")
        
        B, T, N, C = active_tokens[0].shape
        num_active = len(active_tokens)
        
        # Estimate uncertainty for each active modality
        uncertainties = []
        for i, (tokens, is_valid) in enumerate(zip(modality_tokens, valid_mask)):
            if is_valid:
                unc = self.uncertainty_estimators[i](tokens)  # [B, T, 1]
                uncertainties.append(unc)
            else:
                uncertainties.append(torch.zeros(B, T, 1, device=active_tokens[0].device))
        
        uncertainties = torch.cat(uncertainties, dim=-1)  # [B, T, num_modalities]
        
        # Compute adaptive weights (inverse of uncertainty)
        # Lower uncertainty → higher weight
        active_uncertainties = uncertainties[:, :, valid_mask]  # [B, T, num_active]
        weights = F.softmax(-active_uncertainties, dim=-1)  # [B, T, num_active]
        
        # Weighted fusion
        stacked = torch.stack(active_tokens, dim=1)  # [B, num_active, T, N, C]
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_active, T, 1, 1]
        fused = (stacked * weights_expanded).sum(dim=1)  # [B, T, N, C]
        
        # Prepare full weights tensor (including inactive modalities)
        full_weights = torch.zeros(B, T, self.num_modalities, device=fused.device)
        full_weights[:, :, valid_mask] = weights
        
        return fused, full_weights, uncertainties


class SkeletonGuidedAttention(nn.Module):
    """
    Attention mechanism guided by skeletal structure
    Combines standard attention with graph-based skeletal priors
    """
    def __init__(self, embed_dim, num_heads, adjacency_matrix, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.attention = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_norm=True,
        )
        
        # Learnable skeletal prior
        # Convert adjacency matrix to attention bias
        self.register_buffer('skeleton_adj', torch.FloatTensor(adjacency_matrix))
        self.skeleton_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, joint_tokens, pos=None):
        """
        Args:
            joint_tokens: [B, T, num_joints, C]
        Returns:
            output: [B, T, num_joints, C]
        """
        B, T, V, C = joint_tokens.shape
        
        # Reshape for attention
        tokens = rearrange(joint_tokens, 'b t v c -> (b t) v c')
        
        # Standard self-attention
        attn_out = self.attention(tokens, pos=pos)
        
        # Add skeleton-guided features
        # Use adjacency matrix to aggregate neighbor features
        tokens_reshaped = rearrange(attn_out, '(b t) v c -> b t v c', b=B, t=T)
        
        # Graph convolution: aggregate features from connected joints
        skeleton_features = torch.einsum('vw,btwc->btvc', 
                                        self.skeleton_adj, 
                                        tokens_reshaped)
        skeleton_features = self.skeleton_proj(skeleton_features)
        
        # Combine with attention output
        output = tokens_reshaped + 0.1 * skeleton_features  # Weighted residual
        
        return output


class HierarchicalSpatioTemporalBlock(nn.Module):
    """
    Hierarchical modeling: local spatial → temporal → global spatial
    """
    def __init__(self, embed_dim, num_heads, num_joints, adjacency_matrix):
        super().__init__()
        self.num_joints = num_joints
        
        # Local spatial (within modality)
        self.local_spatial_attn = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_norm=True,
        )
        
        # Temporal attention
        self.temporal_attn = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_norm=True,
        )
        
        # Global spatial (across all tokens)
        self.global_spatial_attn = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_norm=True,
        )
        
        # Skeleton-guided attention for joint tokens
        self.skeleton_attn = SkeletonGuidedAttention(
            embed_dim, num_heads, adjacency_matrix, num_joints
        )
        
        # Gate for adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, tokens, Ns):
        """
        Args:
            tokens: [B, T, N_total, C]
            Ns: List of token counts per modality
        Returns:
            output: [B, T, N_total, C]
        """
        B, T, N_total, C = tokens.shape
        
        # 1. Local spatial attention (per modality)
        local_out = self._apply_per_modality_attention(
            tokens, Ns, self.local_spatial_attn
        )
        
        # 2. Temporal attention
        temporal_out = self._apply_temporal_attention(tokens)
        
        # 3. Global spatial attention
        global_out = self._apply_global_spatial_attention(tokens)
        
        # 4. Special processing for joint tokens with skeleton guidance
        joint_enhanced = self._apply_skeleton_guidance(tokens, Ns)
        
        # 5. Adaptive fusion with gating
        combined = torch.cat([local_out, temporal_out, global_out], dim=-1)
        gate = self.fusion_gate(combined)
        
        output = gate * local_out + (1 - gate) * (0.5 * temporal_out + 0.5 * global_out)
        
        # Blend in skeleton-enhanced joint tokens
        output = self._insert_joint_tokens(output, joint_enhanced, Ns)
        
        return output
    
    def _apply_per_modality_attention(self, tokens, Ns, attn_module):
        """Apply attention separately within each modality"""
        B, T, N_total, C = tokens.shape
        outputs = []
        cumsum = 0
        
        for n in Ns:
            if n > 0:
                mod_tokens = tokens[:, :, cumsum:cumsum+n, :]  # [B, T, n, C]
                mod_tokens = rearrange(mod_tokens, 'b t n c -> (b t) n c')
                mod_tokens = attn_module(mod_tokens)
                mod_tokens = rearrange(mod_tokens, '(b t) n c -> b t n c', b=B, t=T)
                outputs.append(mod_tokens)
            cumsum += n
        
        if outputs:
            return torch.cat(outputs, dim=2)
        return tokens
    
    def _apply_temporal_attention(self, tokens):
        """Apply attention across temporal dimension"""
        B, T, N, C = tokens.shape
        tokens = rearrange(tokens, 'b t n c -> (b n) t c')
        tokens = self.temporal_attn(tokens)
        tokens = rearrange(tokens, '(b n) t c -> b t n c', b=B, n=N)
        return tokens
    
    def _apply_global_spatial_attention(self, tokens):
        """Apply attention across all spatial tokens"""
        B, T, N, C = tokens.shape
        tokens = rearrange(tokens, 'b t n c -> (b t) n c')
        tokens = self.global_spatial_attn(tokens)
        tokens = rearrange(tokens, '(b t) n c -> b t n c', b=B, t=T)
        return tokens
    
    def _apply_skeleton_guidance(self, tokens, Ns):
        """Apply skeleton-guided attention to joint tokens"""
        B, T, _, C = tokens.shape
        joint_slices = []
        cumsum = 0
        
        for n in Ns:
            if n > 0:
                # Extract joint tokens (after camera token, before register tokens)
                joint_start = cumsum + 1  # Skip camera token
                joint_end = joint_start + self.num_joints
                joint_tokens = tokens[:, :, joint_start:joint_end, :]
                joint_slices.append(joint_tokens)
            cumsum += n
        
        if joint_slices:
            # Stack and process all joint tokens together
            stacked_joints = torch.stack(joint_slices, dim=1)  # [B, M, T, V, C]
            B, M, T, V, C = stacked_joints.shape
            stacked_joints = rearrange(stacked_joints, 'b m t v c -> (b m) t v c')
            
            # Apply skeleton-guided attention
            enhanced = []
            for i in range(M):
                joints = stacked_joints[i*B:(i+1)*B]  # [B, T, V, C]
                joints = self.skeleton_attn(joints)
                enhanced.append(joints)
            
            enhanced = torch.stack(enhanced, dim=1)  # [B, M, T, V, C]
            return enhanced
        
        return None
    
    def _insert_joint_tokens(self, tokens, joint_enhanced, Ns):
        """Insert enhanced joint tokens back into the token sequence"""
        if joint_enhanced is None:
            return tokens
        
        B, M, T, V, C = joint_enhanced.shape
        new_tokens = tokens.clone()
        cumsum = 0
        mod_idx = 0
        
        for n in Ns:
            if n > 0:
                joint_start = cumsum + 1
                joint_end = joint_start + self.num_joints
                new_tokens[:, :, joint_start:joint_end, :] = joint_enhanced[:, mod_idx]
                mod_idx += 1
            cumsum += n
        
        return new_tokens


class ParallelMultiPathAggregator(nn.Module):
    """
    Parallel processing with multiple paths:
    - Local path: intra-modality features
    - Global path: inter-modality fusion
    - Temporal path: temporal coherence
    Then fuse all paths adaptively
    """
    def __init__(self, embed_dim, num_heads, depth, num_modalities=4):
        super().__init__()
        self.depth = depth
        
        # Three parallel branches
        self.local_branch = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, 
                  qkv_bias=True, qk_norm=True)
            for _ in range(depth)
        ])
        
        self.global_branch = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                  qkv_bias=True, qk_norm=True)
            for _ in range(depth)
        ])
        
        self.temporal_branch = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                  qkv_bias=True, qk_norm=True)
            for _ in range(depth)
        ])
        
        # Adaptive fusion of three branches
        self.branch_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(depth)
        ])
        
    def forward(self, tokens, Ns, layer_idx):
        """
        Process tokens through three parallel branches and fuse
        """
        B, T, N, C = tokens.shape
        
        # Local branch: per-modality processing
        local_out = self._process_local(tokens, Ns, layer_idx)
        
        # Global branch: all tokens together
        global_tokens = rearrange(tokens, 'b t n c -> (b t) n c')
        global_out = self.global_branch[layer_idx](global_tokens)
        global_out = rearrange(global_out, '(b t) n c -> b t n c', b=B, t=T)
        
        # Temporal branch: across time
        temporal_tokens = rearrange(tokens, 'b t n c -> (b n) t c')
        temporal_out = self.temporal_branch[layer_idx](temporal_tokens)
        temporal_out = rearrange(temporal_out, '(b n) t c -> b t n c', b=B, n=N)
        
        # Fuse three branches
        combined = torch.cat([local_out, global_out, temporal_out], dim=-1)
        fused = self.branch_fusion[layer_idx](combined)
        
        # Residual connection
        output = tokens + fused
        
        return output
    
    def _process_local(self, tokens, Ns, layer_idx):
        """Process each modality separately"""
        B, T, N_total, C = tokens.shape
        outputs = []
        cumsum = 0
        
        for n in Ns:
            if n > 0:
                mod_tokens = tokens[:, :, cumsum:cumsum+n, :]
                mod_tokens = rearrange(mod_tokens, 'b t n c -> (b t) n c')
                mod_tokens = self.local_branch[layer_idx](mod_tokens)
                mod_tokens = rearrange(mod_tokens, '(b t) n c -> b t n c', b=B, t=T)
                outputs.append(mod_tokens)
            else:
                # Placeholder for missing modality
                outputs.append(torch.zeros(B, T, 0, C, device=tokens.device))
            cumsum += n
        
        return torch.cat(outputs, dim=2)


class FPNStyleOutputAggregator(nn.Module):
    """
    Feature Pyramid Network style aggregation
    Fuses multi-scale features from different layers
    """
    def __init__(self, embed_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(num_layers)
        ])
        
        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(num_layers - 1)
        ])
        
    def forward(self, feature_pyramid: List[Tensor]):
        """
        Args:
            feature_pyramid: List of [B, M, T, V, C] features from different layers
        Returns:
            multi_scale_outputs: Dict with coarse, medium, fine features
        """
        if len(feature_pyramid) == 0:
            return {}
        
        # Apply lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feature_pyramid)]
        
        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + laterals[i]
        
        # Multi-scale outputs
        outputs = {
            'coarse': feature_pyramid[0] if len(feature_pyramid) > 0 else None,
            'medium': feature_pyramid[len(feature_pyramid) // 2] if len(feature_pyramid) > 1 else None,
            'fine': laterals[-1] if len(laterals) > 0 else None,
        }
        
        return outputs


class TransformerAggregatorV3(nn.Module):
    """
    Improved multimodal multi-view aggregator with:
    1. Parallel multi-path processing
    2. Dynamic modality fusion
    3. Hierarchical spatio-temporal modeling
    4. Skeleton-guided attention
    5. FPN-style output aggregation
    """
    def __init__(self,
                 input_dims=[512, 512, 512, 512],
                 embed_dim=512,
                 num_register_tokens=4,
                 depth=12,
                 num_heads=8,
                 skeleton_type="h36m",
                 use_parallel_paths=True,
                 use_dynamic_fusion=True,
                 use_skeleton_guidance=True,
                 use_fpn_output=True,
                 use_grad_ckpt=False,
                 max_spatial_seq_len=1000,
                 max_temporal_seq_len=243,
                 ):
        super().__init__()

        match skeleton_type:
            case "h36m" | "mmfi":
                skeleton = H36MSkeleton()
            case "smpl":
                skeleton = SMPLSkeleton()
            case _:
                raise ValueError(f"Unknown skeleton type: {skeleton_type}")
        
        self.num_joints = skeleton.num_joints
        self.num_modalities = 4
        self.depth = depth
        self.use_grad_ckpt = use_grad_ckpt
        self.use_parallel_paths = use_parallel_paths
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_skeleton_guidance = use_skeleton_guidance
        self.use_fpn_output = use_fpn_output
        
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
        
        # Dynamic modality fusion
        if use_dynamic_fusion:
            self.modality_fusion = DynamicModalityFusion(embed_dim, self.num_modalities)
        
        # Skeleton-guided processing
        A = get_adjacency_matrix(skeleton.bones, skeleton.num_joints)
        
        if use_parallel_paths:
            # Parallel multi-path architecture
            self.parallel_aggregator = ParallelMultiPathAggregator(
                embed_dim, num_heads, depth, self.num_modalities
            )
        
        if use_skeleton_guidance:
            # Hierarchical spatio-temporal blocks with skeleton guidance
            self.hier_blocks = nn.ModuleList([
                HierarchicalSpatioTemporalBlock(embed_dim, num_heads, self.num_joints, A)
                for _ in range(depth)
            ])
        else:
            # Fallback to standard blocks
            self.blocks = nn.ModuleList([
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                      qkv_bias=True, qk_norm=True)
                for _ in range(depth)
            ])
        
        # GCN blocks for joint refinement
        self.gcn_blocks = nn.ModuleList([
            GCNBlock(embed_dim, embed_dim, A, adaptive=True)
            for _ in range(depth)
        ])
        
        # FPN-style output aggregation
        if use_fpn_output:
            self.output_aggregator = FPNStyleOutputAggregator(embed_dim, depth)
        
        # Final output projection
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
            features: Tuple of (rgb, depth, lidar, mmwave) features
                     Each is [B, T, N, C] or None
        Returns:
            If use_fpn_output:
                Dict with 'coarse', 'medium', 'fine' multi-scale outputs
            Else:
                Final output [B, num_modalities, T, 1+num_joints, C]
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
            raise ValueError("At least one modality must be provided")
        
        # Add positional embeddings - FIXED: per-modality position encoding
        temporal_pe = self.temporal_pos_embed[:T, :].unsqueeze(0).unsqueeze(2)  # [1, T, 1, C]
        
        for i, feat in enumerate(projected_features):
            if feat is not None:
                N = feat.shape[2]
                spatial_pe = self.spatial_pos_embed[:N, :].unsqueeze(0).unsqueeze(0)  # [1, 1, N, C]
                projected_features[i] = feat + spatial_pe + temporal_pe
        
        # Prepare special tokens
        camera_tokens = self.camera_token.expand(B, T, -1, -1)  # [B, T, 1, C]
        joint_tokens = self.joint_tokens.expand(B, T, -1, -1)   # [B, T, V, C]
        register_tokens = self.register_tokens.expand(B, T, -1, -1)  # [B, T, R, C]
        
        # Assemble tokens with special tokens
        modality_tokens = []
        Ns = []
        
        for i, feat in enumerate(projected_features):
            if feat is not None:
                # Concatenate: [camera, joints, registers, patches]
                mod_tokens = torch.cat([
                    camera_tokens, joint_tokens, register_tokens, feat
                ], dim=2)  # [B, T, 1+V+R+N, C]
                
                # Add modality embedding
                mod_tokens = mod_tokens + self.modality_embed[i].view(1, 1, 1, -1)
                modality_tokens.append(mod_tokens)
                Ns.append(mod_tokens.shape[2])
            else:
                Ns.append(0)
        
        # Concatenate all modalities
        tokens = torch.cat(modality_tokens, dim=2)  # [B, T, N_total, C]
        
        # Process through layers
        feature_pyramid = []
        
        for layer_idx in range(self.depth):
            # Option 1: Parallel multi-path processing
            if self.use_parallel_paths:
                tokens = self.parallel_aggregator(tokens, Ns, layer_idx)
            
            # Option 2: Hierarchical spatio-temporal with skeleton guidance
            elif self.use_skeleton_guidance:
                tokens = self.hier_blocks[layer_idx](tokens, Ns)
            
            # Option 3: Standard transformer block
            else:
                B_cur, T_cur, N_cur, C_cur = tokens.shape
                tokens_flat = rearrange(tokens, 'b t n c -> (b t) n c')
                tokens_flat = self.blocks[layer_idx](tokens_flat)
                tokens = rearrange(tokens_flat, '(b t) n c -> b t n c', b=B_cur, t=T_cur)
            
            # Apply GCN refinement on joint tokens
            tokens = self._apply_gcn_refinement(tokens, Ns, layer_idx)
            
            # Extract output tokens for this layer
            output_tokens = self._extract_output_tokens(tokens, Ns)
            feature_pyramid.append(output_tokens)
        
        # Aggregate outputs
        if self.use_fpn_output:
            outputs = self.output_aggregator(feature_pyramid)
            # Project each output
            for key in outputs:
                if outputs[key] is not None:
                    outputs[key] = self.output_proj(outputs[key])
            return outputs
        else:
            # Return final layer output
            final_output = feature_pyramid[-1]
            final_output = self.output_proj(final_output)
            
            if return_intermediates:
                return final_output, feature_pyramid
            return final_output
    
    def _apply_gcn_refinement(self, tokens, Ns, layer_idx):
        """Apply GCN to refine joint tokens"""
        B, T, N_total, C = tokens.shape
        
        # Extract joint tokens from each modality
        joint_slices = []
        joint_positions = []
        cumsum = 0
        
        for i, n in enumerate(Ns):
            if n > 0:
                # Joints are at positions [1:1+num_joints] (after camera token)
                joint_start = cumsum + 1
                joint_end = joint_start + self.num_joints
                joint_slice = tokens[:, :, joint_start:joint_end, :]  # [B, T, V, C]
                joint_slices.append(joint_slice)
                joint_positions.append(joint_start)
            cumsum += n
        
        if not joint_slices:
            return tokens
        
        # Stack and process
        stacked = torch.stack(joint_slices, dim=1)  # [B, M, T, V, C]
        M = stacked.shape[1]
        
        # Reshape for GCN: [BM, C, T, V]
        gcn_input = rearrange(stacked, 'b m t v c -> (b m) c t v')
        
        # Apply GCN
        if self.use_grad_ckpt and self.training:
            gcn_output = checkpoint(self.gcn_blocks[layer_idx], gcn_input, use_reentrant=False)
        else:
            gcn_output = self.gcn_blocks[layer_idx](gcn_input)
        
        # Reshape back
        refined = rearrange(gcn_output, '(b m) c t v -> b m t v c', b=B, m=M)
        
        # Insert back into tokens
        new_tokens = tokens.clone()
        for mod_idx, joint_start in enumerate(joint_positions):
            new_tokens[:, :, joint_start:joint_start + self.num_joints, :] = refined[:, mod_idx]
        
        return new_tokens
    
    def _extract_output_tokens(self, tokens, Ns):
        """
        Extract camera + joint tokens from each modality
        Returns: [B, M, T, 1+V, C]
        """
        B, T, _, C = tokens.shape
        output_slices = []
        cumsum = 0
        
        for n in Ns:
            if n > 0:
                # Extract camera (1) + joints (V)
                output_slice = tokens[:, :, cumsum:cumsum + 1 + self.num_joints, :]
                output_slices.append(output_slice)
            cumsum += n
        
        if output_slices:
            output = torch.stack(output_slices, dim=1)  # [B, M, T, 1+V, C]
        else:
            output = tokens.new_zeros(B, 0, T, 1 + self.num_joints, C)
        
        return output
