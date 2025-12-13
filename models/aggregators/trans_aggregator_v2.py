import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange, reduce, repeat
import math
# from models.video_encoders.layers.rope import PositionGetter3D, RotaryPositionEmbedding3D
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

class TransformerAggregatorV2(nn.Module):
    def __init__(self, 
                 input_dims=[512, 512, 512, 512],
                 embed_dim=512, 
                 num_register_tokens=4, 
                 aa_order=["single", "global", "cross"], 
                 aa_block_size=1, 
                 depth=24, 
                 block_type="Block",
                 skeleton_type="mmfi",
                 num_heads=16,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 proj_bias=True,
                 ffn_bias=True,
                 qk_norm=True,
                 init_values=0.01,
                 use_grad_ckpt=False,
                 max_spatial_seq_len=1000,
                 max_temporal_seq_len=1000,
                 ):
        super(TransformerAggregatorV2, self).__init__()

        # Skeleton type
        match skeleton_type:
            case "h36m" | "mmfi":
                skeleton = H36MSkeleton()
            case "smpl":
                skeleton = SMPLSkeleton()
            case _:
                raise ValueError(f"Unknown skeleton type: {skeleton_type}")

        self.num_joints = skeleton.num_joints

        # Input projection layers
        rgb_dim, depth_dim, lidar_dim, mmwave_dim = input_dims
        self.proj_rgb = nn.Sequential(
            nn.Linear(rgb_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim)
        )
        self.proj_depth = nn.Sequential(
            nn.Linear(depth_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim)
        )
        self.proj_lidar = nn.Sequential(
            nn.Linear(lidar_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim)
        )
        self.proj_mmwave = nn.Sequential(
            nn.Linear(mmwave_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim)
        )

        # Special tokens
        self.camera_token = nn.Parameter(torch.randn(1, 1, 2, 1, embed_dim)) # B, T, (Anchor or not), N, D
        self.joint_token = nn.Parameter(torch.randn(1, 1, 2, self.num_joints, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 1, 2, num_register_tokens, embed_dim))
        self.num_register_tokens = num_register_tokens

        # Embeddings
        self.modality_embed = nn.Parameter(torch.randn(1, 1, 4, embed_dim)) # B, T, M, D
        # self.token_type_embed = nn.Parameter(torch.randn(1, 1, 4, embed_dim)) # B, T, (Camera, Joint, Register, Patch), D
        self.register_buffer("spatial_pos_embed", get_sinusoidal_embeddings(
            max_spatial_seq_len, embed_dim, device='cpu'), persistent=False)
        self.register_buffer("temporal_pos_embed", get_sinusoidal_embeddings(
            max_temporal_seq_len, embed_dim, device='cpu'), persistent=False)

        match block_type:
            case "Block":
                block_cls = Block
            case _:
                raise ValueError(f"Unknown block type: {block_type}")
            
        block_params = dict(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            init_values=init_values,
        )

        if 'single' in aa_order:
            self.single_blocks = nn.ModuleList([block_cls(**block_params) for _ in range(depth)])
        if 'global' in aa_order:
            self.global_blocks = nn.ModuleList([block_cls(**block_params) for _ in range(depth)])
        if 'spatial' in aa_order:
            self.spatial_blocks = nn.ModuleList([block_cls(**block_params) for _ in range(depth)])
        if 'temporal' in aa_order:
            self.temporal_blocks = nn.ModuleList([block_cls(**block_params) for _ in range(depth)])
        if 'cross_2d_3d' in aa_order:
            self.cross_2d_3d_blocks = nn.ModuleList([CABlock(**block_params) for _ in range(depth)])
        if 'cross_token' in aa_order:
            self.cross_token_blocks = nn.ModuleList([CABlock(**block_params) for _ in range(depth)])

        A = get_adjacency_matrix(skeleton.bones, skeleton.num_joints)

        if 'gcn' in aa_order:
            self.gcn_blocks = nn.ModuleList([
                GCNBlock(embed_dim, embed_dim, A, adaptive=True) for _ in range(depth)
            ])
        
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.depth = depth
        self.aa_block_num = depth // aa_block_size

        self.use_grad_ckpt = use_grad_ckpt

        self.anchor_map = {
            'input_rgb': 0,
            'input_depth': 1,
            'input_lidar': 2,
            'input_mmwave': 3
        }

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features, **kwargs):
        # B, T, Nx, Cx for each modality
        features_rgb, features_depth, features_lidar, features_mmwave = features

        B, T, _, _ = 0, 0, 0, 0
        if features_rgb is not None:
            features_rgb = self.proj_rgb(features_rgb)
            B, T, _, _ = features_rgb.shape
        if features_depth is not None:
            features_depth = self.proj_depth(features_depth)
            B, T, _, _ = features_depth.shape
        if features_lidar is not None:
            features_lidar = self.proj_lidar(features_lidar)
            B, T, _, _ = features_lidar.shape
        if features_mmwave is not None:
            features_mmwave = self.proj_mmwave(features_mmwave)
            B, T, _, _ = features_mmwave.shape
        if B == 0:
            raise ValueError("At least one modality must be provided.")
        
        # Add positional embeddings
        spatial_pos_embed = self.spatial_pos_embed[:max(
            features_rgb.shape[2] if features_rgb is not None else 0,
            features_depth.shape[2] if features_depth is not None else 0,
            features_lidar.shape[2] if features_lidar is not None else 0,
            features_mmwave.shape[2] if features_mmwave is not None else 0,
        ), :].unsqueeze(0).unsqueeze(0)  # 1, 1, N, D
        temporal_pos_embed = self.temporal_pos_embed[:T, :].unsqueeze(0).unsqueeze(2)  # 1, T, 1, D
        if features_rgb is not None:
            features_rgb = features_rgb + spatial_pos_embed[:, :, :features_rgb.shape[2], :] + temporal_pos_embed
        if features_depth is not None:
            features_depth = features_depth + spatial_pos_embed[:, :, :features_depth.shape[2], :] + temporal_pos_embed
        if features_lidar is not None:
            features_lidar = features_lidar + spatial_pos_embed[:, :, :features_lidar.shape[2], :] + temporal_pos_embed
        if features_mmwave is not None:
            features_mmwave = features_mmwave + spatial_pos_embed[:, :, :features_mmwave.shape[2], :] + temporal_pos_embed
        
        # Expand special tokens
        camera_tokens_normal, camera_tokens_anchor = self.expand_special_tokens(self.camera_token, B, T)
        joint_tokens_normal, joint_tokens_anchor = self.expand_special_tokens(self.joint_token, B, T)
        register_tokens_normal, register_tokens_anchor = self.expand_special_tokens(self.register_token, B, T)

        # Concatenate special tokens with patch tokens
        anchor_idx = self.anchor_map.get(kwargs.get('anchor_key', None), -1)

        features_list = []
        Ns = []
        for i, feat in enumerate([features_rgb, features_depth, features_lidar, features_mmwave]):
            if feat is not None:
                if i == anchor_idx:
                    feat = self.insert_special_tokens(feat, camera_tokens_anchor, joint_tokens_anchor, register_tokens_anchor)
                else:
                    feat = self.insert_special_tokens(feat, camera_tokens_normal, joint_tokens_normal, register_tokens_normal)
                features_list.append(feat)
                Ns.append(feat.shape[2])
            else:
                Ns.append(0)

        # Add modality embeddings
        for i, feat in enumerate(features_list):
            feat = feat + self.modality_embed[:, :, i:i+1, :].unsqueeze(2)  # B, T, N, D
            features_list[i] = feat

        tokens = torch.cat(features_list, dim=2)  # B, T, N_total, D

        single_idx, global_idx, spatial_idx, temporal_idx, cross_2d_3d_idx, cross_token_idx, gcn_idx = 0, 0, 0, 0, 0, 0, 0
        output_list = []
        for _ in range(self.aa_block_num):

            output_per_block = []
            for aa_type in self.aa_order:
                match aa_type:
                    case "single":
                        tokens, single_idx, intermediates = self._process_single_attention(
                            tokens, Ns, single_idx, pos=None)
                    case "global":
                        tokens, global_idx, intermediates = self._process_global_attention(
                            tokens, Ns, global_idx, pos=None)
                    case "spatial":
                        tokens, spatial_idx, intermediates = self._process_spatial_attention(
                            tokens, Ns, spatial_idx, pos=None)
                    case "temporal":
                        tokens, temporal_idx, intermediates = self._process_temporal_attention(
                            tokens, Ns, temporal_idx, pos=None)
                    case "cross_2d_3d":
                        tokens, cross_2d_3d_idx, intermediates = self._process_cross_2d_3d_attention(
                            tokens, Ns, cross_2d_3d_idx, pos=None)
                    case "cross_token":
                        tokens, cross_token_idx, intermediates = self._process_cross_token_attention(
                            tokens, Ns, cross_token_idx, pos=None)
                    case "gcn":
                        tokens, gcn_idx, intermediates = self._process_gcn(
                            tokens, Ns, gcn_idx, pos=None)
                    case _:
                        raise ValueError(f"Unknown attention type: {aa_type}")
                output_per_block.extend(intermediates)
            output_per_block = torch.sum(torch.stack(output_per_block), dim=0)
            output_list.append(output_per_block)
        
        return output_list

    def _process_single_attention(self, tokens, Ns, idx, pos=None):
        # Process single-attention blocks
        # tokens: B, T, N_total, C
        B, T, N_total, C = tokens.shape
        N_cumsum = torch.tensor([0] + list(Ns), device=tokens.device).cumsum(0)
        intermediates = []
        # work on a non-view base tensor to avoid inplace issues on views
        tokens_base = tokens

        for _ in range(self.aa_block_size):
            updated_slices = []
            # Process modalities separately to avoid in-place modification on views
            for i in range(4):
                start, end = N_cumsum[i].item(), N_cumsum[i+1].item()
                if end > start:
                    tokens_slice = tokens_base[:, :, start:end, :].reshape(B, T * (end - start), C)

                    if self.use_grad_ckpt and self.training:
                        tokens_slice = checkpoint(self.single_blocks[idx], tokens_slice, pos, use_reentrant=False)
                    else:
                        tokens_slice = self.single_blocks[idx](tokens_slice, pos=pos)

                    updated_slices.append((i, tokens_slice.reshape(B, T, end - start, C)))

            # reconstruct tokens from updated slices to avoid inplace writes into views
            new_tokens = tokens_base.clone()
            for i, slice_tensor in updated_slices:
                start, end = N_cumsum[i].item(), N_cumsum[i+1].item()
                new_tokens[:, :, start:end, :] = slice_tensor

            tokens_base = new_tokens
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base, Ns))

        return tokens_base, idx, intermediates
    
    def _process_global_attention(self, tokens, Ns, idx, pos=None):
        # Process global-attention blocks
        B, T, N_total, C = tokens.shape
        intermediates = []
        tokens_base = tokens

        for _ in range(self.aa_block_size):
            tokens_reshaped = tokens_base.reshape(B, T * N_total, C)

            if self.use_grad_ckpt and self.training:
                tokens_reshaped = checkpoint(self.global_blocks[idx], tokens_reshaped, pos, use_reentrant=False)
            else:
                tokens_reshaped = self.global_blocks[idx](tokens_reshaped, pos=pos)

            tokens_base = tokens_reshaped.reshape(B, T, N_total, C)
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base, Ns))

        return tokens_base, idx, intermediates
    
    def _process_cross_2d_3d_attention(self, tokens, Ns, idx, pos=None):
        # Process cross-attention blocks
        B, T, N_total, C = tokens.shape
        N_2d = Ns[0] + Ns[1]
        N_3d = Ns[2] + Ns[3]
        intermediates = []
        tokens_base = tokens

        for _ in range(self.aa_block_size):
            tokens_2d = tokens_base[:, :, :N_2d, :].reshape(B, T * N_2d, C)
            tokens_3d = tokens_base[:, :, N_2d:, :].reshape(B, T * N_3d, C)

            if self.use_grad_ckpt and self.training:
                tokens_2d = checkpoint(self.cross_blocks[idx], tokens_2d, tokens_3d, pos, use_reentrant=False)
                tokens_3d = checkpoint(self.cross_blocks[idx], tokens_3d, tokens_2d, pos, use_reentrant=False)
            else:
                tokens_2d_new = self.cross_blocks[idx](tokens_2d, context=tokens_3d, pos=pos)
                tokens_3d_new = self.cross_blocks[idx](tokens_3d, context=tokens_2d, pos=pos)
                tokens_2d, tokens_3d = tokens_2d_new, tokens_3d_new

            new_tokens = tokens_base.clone()
            new_tokens[:, :, :N_2d, :] = tokens_2d.reshape(B, T, N_2d, C)
            new_tokens[:, :, N_2d:, :] = tokens_3d.reshape(B, T, N_3d, C)

            tokens_base = new_tokens
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base, Ns))

        return tokens_base, idx, intermediates
    
    def _process_spatial_attention(self, tokens, Ns, idx, pos=None):
        # Process spatial-attention blocks
        B, T, N_total, C = tokens.shape
        intermediates = []
        tokens_base = tokens

        for _ in range(self.aa_block_size):
            tokens_reshaped = rearrange(tokens_base, 'b t n c -> (b t) n c')

            if self.use_grad_ckpt and self.training:
                tokens_reshaped = checkpoint(self.spatial_blocks[idx], tokens_reshaped, pos, use_reentrant=False)
            else:
                tokens_reshaped = self.spatial_blocks[idx](tokens_reshaped, pos=pos)

            tokens_base = rearrange(tokens_reshaped, '(b t) n c -> b t n c', b=B, t=T)
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base, Ns))

        return tokens_base, idx, intermediates
    
    def _process_temporal_attention(self, tokens, Ns, idx, pos=None):
        # Process temporal-attention blocks
        B, T, N_total, C = tokens.shape
        intermediates = []
        tokens_base = tokens

        for _ in range(self.aa_block_size):
            tokens_reshaped = rearrange(tokens_base, 'b t n c -> (b n) t c')

            if self.use_grad_ckpt and self.training:
                tokens_reshaped = checkpoint(self.temporal_blocks[idx], tokens_reshaped, pos, use_reentrant=False)
            else:
                tokens_reshaped = self.temporal_blocks[idx](tokens_reshaped, pos=pos)

            tokens_base = rearrange(tokens_reshaped, '(b n) t c -> b t n c', b=B, n=N_total)
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base, Ns))

        return tokens_base, idx, intermediates
    
    def _process_cross_token_attention(self, tokens, Ns, idx, pos=None):
        """
        Cross-attention between camera+joint tokens (query) and all tokens (key/value).
        This allows camera and joint tokens to aggregate information from patch tokens
        and register tokens across all modalities.
        """
        B, T, N_total, C = tokens.shape

        # N_total = (1+num_joints+num_register_tokens+N_rgb) + 
        #           (1+num_joints+num_register_tokens+N_depth) + 
        #           (1+num_joints+num_register_tokens+N_lidar) + 
        #           (1+num_joints+num_register_tokens+N_mmwave)

        N_camera = 1
        N_joint = self.num_joints
        N_register = self.num_register_tokens
        N_special = N_camera + N_joint  # We only use camera + joint as queries
        
        intermediates = []
        tokens_base = tokens
        
        for _ in range(self.aa_block_size):
            # Extract query tokens (camera + joint from each modality)
            query_slices = []
            cumsum = 0
            
            for i, n in enumerate(Ns):
                if n > 0:
                    # Extract camera (1) + joint (num_joints) tokens from this modality
                    query_slice = tokens_base[:, :, cumsum:cumsum + N_special, :]
                    query_slices.append(query_slice)
                cumsum += n
            
            if len(query_slices) == 0:
                # No modalities present, skip
                idx += 1
                intermediates.append(self._extract_output_tokens(tokens_base, Ns))
                continue
            
            # Stack queries: [B, T, M*N_special, C] where M = num active modalities
            queries = torch.cat(query_slices, dim=2)  # B, T, M*N_special, C
            queries = queries.reshape(B, T * queries.shape[2], C)
            
            # Use all tokens as key/value context
            context = tokens_base.reshape(B, T * N_total, C)
            
            # Apply cross-attention
            if self.use_grad_ckpt and self.training:
                queries_updated = checkpoint(
                    self.cross_token_blocks[idx], 
                    queries, 
                    context, 
                    pos, 
                    use_reentrant=False
                )
            else:
                queries_updated = self.cross_token_blocks[idx](
                    queries, 
                    context=context, 
                    pos=pos
                )
            
            # Reshape back and insert into original positions
            queries_updated = queries_updated.reshape(B, T, -1, C)  # B, T, M*N_special, C
            
            new_tokens = tokens_base.clone()
            cumsum = 0
            query_idx = 0
            
            for i, n in enumerate(Ns):
                if n > 0:
                    # Put updated camera + joint tokens back
                    new_tokens[:, :, cumsum:cumsum + N_special, :] = \
                        queries_updated[:, :, query_idx:query_idx + N_special, :]
                    query_idx += N_special
                cumsum += n
            
            tokens_base = new_tokens
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base, Ns))
        
        return tokens_base, idx, intermediates
        
    def _process_gcn(self, tokens, Ns, idx, pos=None):
        """
        Process GCN blocks on joint tokens.
        Extract joint tokens, apply ST-GCN, and reinsert them back.
        """
        B, T, N_total, C = tokens.shape
        N_camera = 1
        N_joint = self.num_joints
        N_register = self.num_register_tokens
        
        intermediates = []
        tokens_base = tokens
        
        for _ in range(self.aa_block_size):
            # Extract joint tokens from each modality
            joint_slices = []
            joint_positions = []  # Track (modality_idx, start_pos)
            cumsum = 0
            
            for i, n in enumerate(Ns):
                if n > 0:
                    # Joint tokens are at position [cumsum + N_camera : cumsum + N_camera + N_joint]
                    joint_start = cumsum + N_camera
                    joint_end = joint_start + N_joint
                    joint_slice = tokens_base[:, :, joint_start:joint_end, :]  # B, T, N_joint, C
                    joint_slices.append(joint_slice)
                    joint_positions.append((i, joint_start))
                cumsum += n
            
            if len(joint_slices) == 0:
                # No modalities present, skip
                idx += 1
                intermediates.append(self._extract_output_tokens(tokens_base, Ns))
                continue
            
            # Stack joint tokens: [B, M, T, N_joint, C] where M = num active modalities
            joint_tokens = torch.stack(joint_slices, dim=1)  # B, M, T, N_joint, C
            
            # Reshape for GCN: [B*M, C, T, N_joint] (GCN expects N, C, T, V format)
            M = joint_tokens.shape[1]
            joint_tokens = rearrange(joint_tokens, 'b m t v c -> (b m) c t v')
            
            # Apply GCN
            if self.use_grad_ckpt and self.training:
                joint_tokens = checkpoint(
                    self.gcn_blocks[idx],
                    joint_tokens,
                    use_reentrant=False
                )
            else:
                joint_tokens = self.gcn_blocks[idx](joint_tokens)
            
            # Reshape back: [B, M, T, N_joint, C]
            joint_tokens = rearrange(joint_tokens, '(b m) c t v -> b m t v c', b=B, m=M)
            
            # Reinsert joint tokens back into original positions
            new_tokens = tokens_base.clone()
            for mod_idx, (i, joint_start) in enumerate(joint_positions):
                new_tokens[:, :, joint_start:joint_start + N_joint, :] = joint_tokens[:, mod_idx]
            
            tokens_base = new_tokens
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base, Ns))
        
        return tokens_base, idx, intermediates

    def _expand_special_tokens(self, tokens, B, T):
        tokens = tokens.expand(B, T, -1, -1, -1)  # B, T, 2, N, C
        tokens_normal, tokens_anchor = tokens[:, :, 0], tokens[:, :, 1]  # B, T, N, C
        return tokens_normal, tokens_anchor
    
    def _insert_special_tokens(self, features, camera_tokens, joint_tokens, register_tokens):
        if features is not None:
            features = torch.cat([camera_tokens, joint_tokens, register_tokens, features], dim=2)
        return features
    
    def _extract_output_tokens(self, tokens, Ns):
        # Extract output tokens without relying on views that are later modified in-place
        B, T, _, C = tokens.shape

        token_slices = []
        cumsum = 0
        for n in Ns:
            if n > 0:
                # Take the first 2 tokens of each modality and clone to break view relationship
                token_slices.append(tokens[:, :, cumsum:cumsum + 1 + self.joints, :].contiguous())
            cumsum += n

        if len(token_slices) == 0:
            # Handle edge case where all Ns are zero
            return tokens.new_zeros(B, 0, T, 1 + self.joints, C)

        output = torch.stack(token_slices, dim=1)  # B, M, T, 1 + self.joints, C
        return output
