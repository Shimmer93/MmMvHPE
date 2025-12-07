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
                 use_grad_ckpt=False,
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

        self.use_grad_ckpt = use_grad_ckpt

        self.anchor_map = {
            'input_rgb': 0,
            'input_depth': 1,
            'input_lidar': 2,
            'input_mmwave': 3
        }

        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        nn.init.normal_(self.trainable_cond_mask.weight, std=1e-6)


    def forward(self, features, **kwargs):
        # Forward pass implementation
        
        # B, T, Nx, Cx for each modality
        features_rgb, features_depth, features_lidar, features_mmwave = features

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

        # # Add RoPE to RGB and Depth features if applicable
        # if self.rope is not None:
        #     pos = self.position_getter(B, T, features_rgb.shape[2], features_rgb.shape[3], features_rgb.device)

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
        
        features_cat = torch.cat(features_list, dim=2)  # B, T, N_total, C

        N_cumsum = torch.tensor([0] + Ns, device=features_cat.device).cumsum(0)
        mask_input = torch.zeros(B, T, N_cumsum[-1], dtype=torch.long, device=features_cat.device)
        for i in range(1, 4):  # depth, lidar, mmwave
            if Ns[i] > 0:
                mask_input[:, :, N_cumsum[i]:N_cumsum[i+1]] = i

        # mask_input = torch.zeros([B, T, N_total], dtype=torch.long, device=features_cat.device)
        # if N_depth > 0:
        #     mask_input[:, :, N_rgb:N_rgb + N_depth] = 1
        # if N_lidar > 0:
        #     mask_input[:, :, N_rgb + N_depth:N_rgb + N_depth + N_lidar] = 2
        # if N_mmwave > 0:
        #     mask_input[:, :, N_rgb + N_depth + N_lidar:] = 3

        cond_mask = self.trainable_cond_mask(mask_input)  # B, T, N_total, C
        tokens = features_cat + cond_mask
        del features_cat, cond_mask  # Free memory

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
                output_list.append(
                    torch.cat([single_intermediates[i], global_intermediates[i], cross_intermediates[i]], dim=-1)
            )

            del single_intermediates, global_intermediates, cross_intermediates
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            intermediates.append(self.extract_output_tokens(tokens_base, Ns))

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
            intermediates.append(self.extract_output_tokens(tokens_base, Ns))

        return tokens_base, idx, intermediates
    
    def _process_cross_attention(self, tokens, Ns, idx, pos=None):
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
            intermediates.append(self.extract_output_tokens(tokens_base, Ns))

        return tokens_base, idx, intermediates

    def expand_special_tokens(self, tokens, B, T):
        tokens = tokens.expand(B, T, -1, -1, -1)  # B, T, 2, N, C
        tokens_normal, tokens_anchor = tokens[:, :, 0], tokens[:, :, 1]  # B, T, N, C
        return tokens_normal, tokens_anchor
    
    def insert_special_tokens(self, features, camera_tokens, joint_tokens, register_tokens):
        if features is not None:
            features = torch.cat([camera_tokens, joint_tokens, register_tokens, features], dim=2)
        return features
    
    def extract_output_tokens(self, tokens, Ns):
        # Extract output tokens without relying on views that are later modified in-place
        B, T, _, C = tokens.shape

        token_slices = []
        cumsum = 0
        for n in Ns:
            if n > 0:
                # Take the first 2 tokens of each modality and clone to break view relationship
                token_slices.append(tokens[:, :, cumsum:cumsum + 2, :].contiguous())
            cumsum += n

        if len(token_slices) == 0:
            # Handle edge case where all Ns are zero
            return tokens.new_zeros(B, 0, T, 2, C)

        output = torch.stack(token_slices, dim=1)  # B, M, T, 2, C
        return output