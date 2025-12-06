import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange, reduce, repeat
# from models.video_encoders.layers.rope import PositionGetter3D, RotaryPositionEmbedding3D
from models.video_encoders.layers.block import Block
from .layers.block import CABlock

class TransformerAggregator(nn.Module):
    def __init__(self, 
                 input_dims=[512, 512, 512, 128],
                 embed_dim=512, 
                 num_register_tokens=4, 
                 aa_order=["single", "global", "cross"], 
                 aa_block_size=1, 
                 depth=24, 
                 block_type="Block",
                 num_heads=16,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 proj_bias=True,
                 ffn_bias=True,
                 qk_norm=True,
                 init_values=0.01,
                 ):
        super(TransformerAggregator, self).__init__()

        rgb_dim, depth_dim, lidar_dim, mmwave_dim = input_dims
        self.proj_rgb = nn.Linear(rgb_dim, embed_dim)
        self.proj_depth = nn.Linear(depth_dim, embed_dim)
        self.proj_lidar = nn.Linear(lidar_dim, embed_dim)
        self.proj_mmwave = nn.Linear(mmwave_dim, embed_dim)

        # Implementation of the Transformer-based aggregator goes here
        self.camera_token = nn.Parameter(torch.randn(1, 1, 2, 1, embed_dim))
        self.joint_token = nn.Parameter(torch.randn(1, 1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 1, 2, num_register_tokens, embed_dim))
        self.trainable_cond_mask = nn.Embedding(4, embed_dim)

        # self.rope = RotaryPositionEmbedding3D(embed_dim, freq=rope_freq) if rope_freq > 0 else None
        # self.position_getter = PositionGetter3D() if rope_freq > 0 else None

        if block_type == "Block":
            block_fn = Block
        else:
            raise NotImplementedError(f"Block type {block_type} not implemented.")

        if "single" in aa_order:
            self.single_blocks = nn.ModuleList([
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                ) for _ in range(depth)
            ])

        if "global" in aa_order:
            self.global_blocks = nn.ModuleList([
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                ) for _ in range(depth)
            ])

        if "cross" in aa_order:
            self.cross_blocks = nn.ModuleList([
                CABlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                ) for _ in range(depth)
            ])

        self.depth = depth
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.aa_block_num = depth // aa_block_size
        self.patch_start_idx = 1 + num_register_tokens

        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        nn.init.normal_(self.trainable_cond_mask.weight, std=1e-6)


    def forward(self, features, **kwargs):
        # Forward pass implementation
        
        # B, T, Nx, Cx for each modality
        features_rgb, features_depth, features_lidar, features_mmwave = features

        if isinstance(features_rgb, dict):
            features_rgb = features_rgb['x_norm_patchtokens']
        if isinstance(features_depth, dict):
            features_depth = features_depth['x_norm_patchtokens']

        if features_rgb is not None:
            features_rgb = self.proj_rgb(features_rgb)
        if features_depth is not None:
            features_depth = self.proj_depth(features_depth)
        if features_lidar is not None:
            features_lidar = self.proj_lidar(features_lidar)
        if features_mmwave is not None:
            features_mmwave = self.proj_mmwave(features_mmwave)

        B, T = 0, 0
        for feat in [features_rgb, features_depth, features_lidar, features_mmwave]:
            if feat is not None:
                B, T = feat.shape[0], feat.shape[1]
                break

        # # Add RoPE to RGB and Depth features if applicable
        # if self.rope is not None:
        #     pos = self.position_getter(B, T, features_rgb.shape[2], features_rgb.shape[3], features_rgb.device)

        # Expand special tokens
        camera_tokens_normal, camera_tokens_anchor = self.expand_special_tokens(self.camera_token, B, T)
        joint_tokens_normal, joint_tokens_anchor = self.expand_special_tokens(self.joint_token, B, T)
        register_tokens_normal, register_tokens_anchor = self.expand_special_tokens(self.register_token, B, T)

        # Concatenate special tokens with patch tokens
        anchor_key = kwargs.get('anchor_key', None)
        anchor_map = {
            'input_rgb': 0,
            'input_depth': 1,
            'input_lidar': 2,
            'input_mmwave': 3
        }
        anchor_idx = anchor_map.get(anchor_key, -1)

        features_list = [features_rgb, features_depth, features_lidar, features_mmwave]
        for i, feat in enumerate(features_list):
            if i == anchor_idx:
                features_list[i] = self.insert_special_tokens(feat, camera_tokens_anchor, joint_tokens_anchor, register_tokens_anchor)
            else:
                features_list[i] = self.insert_special_tokens(feat, camera_tokens_normal, joint_tokens_normal, register_tokens_normal)

        features_list = []
        for feat in [features_rgb, features_depth, features_lidar, features_mmwave]:
            if feat is not None:
                features_list.append(feat)
        features_cat = torch.cat(features_list, dim=2)  # B, T, N_total, C

        N_rgb = features_rgb.shape[2] if features_rgb is not None else 0
        N_depth = features_depth.shape[2] if features_depth is not None else 0
        N_lidar = features_lidar.shape[2] if features_lidar is not None else 0
        N_mmwave = features_mmwave.shape[2] if features_mmwave is not None else 0
        Ns = [N_rgb, N_depth, N_lidar, N_mmwave]

        mask_input = torch.zeros([B, T, N_rgb + N_depth + N_lidar + N_mmwave], dtype=torch.long, device=camera_tokens_normal.device)
        mask_input[:, :, N_rgb:N_rgb + N_depth] += 1
        mask_input[:, :, N_rgb + N_depth:N_rgb + N_depth + N_lidar] += 2
        mask_input[:, :, N_rgb + N_depth + N_lidar:] += 3

        cond_mask = self.trainable_cond_mask(mask_input)  # B, T, N_total, C
        tokens = features_cat + cond_mask

        pos = None  # Placeholder for positional encoding if needed

        single_idx = 0
        global_idx = 0
        cross_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for aa_type in self.aa_order:
                if aa_type == "single":
                    tokens, single_idx, single_intermediates = self._process_single_attention(
                        tokens, Ns, single_idx, pos=pos
                    )
                elif aa_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, Ns, global_idx, pos=pos
                    )
                elif aa_type == "cross":
                    tokens, cross_idx, cross_intermediates = self._process_cross_attention(
                        tokens, Ns, cross_idx, pos=pos
                    )
            for i in range(len(single_intermediates)):
                concat_inter = torch.cat(
                    [single_intermediates[i], global_intermediates[i], cross_intermediates[i]], dim=-1
                )  # B, M, T, N_total, C_concat
                output_list.append(concat_inter)

        del concat_inter, single_intermediates, global_intermediates, cross_intermediates

        return output_list
    
    def _process_single_attention(self, tokens, Ns, idx, pos=None):
        # Process single-attention blocks
        # tokens: B, T, N_total, C
        B, T, N_total, C = tokens.shape
        N_rgb, N_depth, N_lidar, N_mmwave = Ns
        intermediates = []
        for _ in range(self.aa_block_size):
            modality_ranges = [
                (0, N_rgb),
                (N_rgb, N_rgb + N_depth),
                (N_rgb + N_depth, N_rgb + N_depth + N_lidar),
                (N_rgb + N_depth + N_lidar, N_rgb + N_depth + N_lidar + N_mmwave)
            ]
            
            for start, end in modality_ranges:
                if end > start:
                    tokens_slice = tokens[:, :, start:end, :].reshape(B, T * (end - start), C)
                    if self.training:
                        tokens_slice = checkpoint(self.single_blocks[idx], tokens_slice, pos=pos, use_reentrant=False)
                    else:
                        tokens_slice = self.single_blocks[idx](tokens_slice, pos=pos)
                    tokens[:, :, start:end, :] = tokens_slice.reshape(B, T, end - start, C)
            
            idx += 1
            intermediates.append(self.extract_output_tokens(tokens, Ns))

        return tokens, idx, intermediates
    
    def _process_global_attention(self, tokens, Ns, idx, pos=None):
        # Process global-attention blocks
        B, T, N_total, C = tokens.shape
        intermediates = []
        for _ in range(self.aa_block_size):
            tokens = tokens.reshape(B, T * N_total, C)
            if self.training:
                tokens = checkpoint(self.global_blocks[idx], tokens, pos=pos, use_reentrant=False)
            else:
                tokens = self.global_blocks[idx](tokens, pos=pos)
            tokens = tokens.reshape(B, T, N_total, C)
            idx += 1
            intermediates.append(self.extract_output_tokens(tokens, Ns))

        return tokens, idx, intermediates
    
    def _process_cross_attention(self, tokens, Ns, idx, pos=None):
        # Process cross-attention blocks
        B, T, N_total, C = tokens.shape
        N_rgb, N_depth, N_lidar, N_mmwave = Ns
        N_2d = N_rgb + N_depth
        N_3d = N_lidar + N_mmwave
        intermediates = []
        for _ in range(self.aa_block_size):
            tokens_2d = tokens[:, :, :N_2d, :].reshape(B, T * N_2d, C)
            tokens_3d = tokens[:, :, N_2d:, :].reshape(B, T * N_3d, C)

            if self.training:
                tokens_2d = checkpoint(
                    self.cross_blocks[idx],
                    tokens_2d,
                    context=tokens_3d,
                    pos=pos,
                    use_reentrant=False,
                )
                tokens_3d = checkpoint(
                    self.cross_blocks[idx],
                    tokens_3d,
                    context=tokens_2d,
                    pos=pos,
                    use_reentrant=False,
                )
            else:
                tokens_2d = self.cross_blocks[idx](tokens_2d, context=tokens_3d, pos=pos)
                tokens_3d = self.cross_blocks[idx](tokens_3d, context=tokens_2d, pos=pos)

            tokens[:, :, :N_2d, :] = tokens_2d.reshape(B, T, N_2d, C)
            tokens[:, :, N_2d:, :] = tokens_3d.reshape(B, T, N_3d, C)

            idx += 1
            intermediates.append(self.extract_output_tokens(tokens, Ns))

        return tokens, idx, intermediates

    def expand_special_tokens(self, tokens, B, T):
        tokens = tokens.expand(B, T, -1, -1, -1)  # B, T, 2, N, C
        tokens_normal, tokens_anchor = tokens[:, :, 0], tokens[:, :, 1]  # B, T, N, C
        return tokens_normal, tokens_anchor
    
    def insert_special_tokens(self, features, camera_tokens, joint_tokens, register_tokens):
        if features is not None:
            features = torch.cat([camera_tokens, joint_tokens, register_tokens, features], dim=2)
        return features
    
    def extract_output_tokens(self, tokens, Ns):
        # Extract output tokens
        N_rgb, N_depth, N_lidar, N_mmwave = Ns
        tokens_list = []
        tokens_rgb = tokens[:, :, :2, :] if N_rgb > 0 else None
        tokens_depth = tokens[:, :, N_rgb:N_rgb + 2, :] if N_depth > 0 else None
        tokens_lidar = tokens[:, :, N_rgb + N_depth:N_rgb + N_depth + 2, :] if N_lidar > 0 else None
        tokens_mmwave = tokens[:, :, N_rgb + N_depth + N_lidar:N_rgb + N_depth + N_lidar + 2, :] if N_mmwave > 0 else None
        for t in [tokens_rgb, tokens_depth, tokens_lidar, tokens_mmwave]:
            if t is not None:
                tokens_list.append(t)
        output = torch.stack(tokens_list, dim=1)  # B, M, T, 2, C

        return output