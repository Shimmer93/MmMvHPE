import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.video_encoders.layers.block import Block
from .layers.block import CABlock


class TransformerAggregatorWithRayPos(nn.Module):
    """
    Transformer aggregator that injects ray-based position encodings for image patch tokens.
    Rays are computed in world coordinates from camera intrinsics/extrinsics and patch centers.
    """

    def __init__(
        self,
        input_dims: List[int] = [512, 512, 512, 512],
        embed_dim: int = 512,
        num_register_tokens: int = 4,
        aa_order: List[str] = ["single", "global", "cross"],
        aa_block_size: int = 1,
        depth: int = 24,
        block_type: str = "Block",
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 0.01,
        use_grad_ckpt: bool = False,
        max_seq_len: int = 1000,
        use_ray_pos: bool = True,
        require_ray_pos: bool = False,
        ray_pos_dim: int = 6,
        normalize_ray_pos: bool = False,
        use_ray_pos_gate: bool = False,
        ray_pos_gate_init: float = 0.0,
        ray_modalities: Optional[List[str]] = None,
    ):
        super().__init__()

        if depth % aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        rgb_dim, depth_dim, lidar_dim, mmwave_dim = input_dims
        self.proj_rgb = nn.Linear(rgb_dim, embed_dim, bias=proj_bias)
        self.proj_depth = nn.Linear(depth_dim, embed_dim, bias=proj_bias)
        self.proj_lidar = nn.Linear(lidar_dim, embed_dim, bias=proj_bias)
        self.proj_mmwave = nn.Linear(mmwave_dim, embed_dim, bias=proj_bias)

        self.camera_token = nn.Parameter(torch.randn(1, 1, 2, 1, embed_dim))
        self.joint_token = nn.Parameter(torch.randn(1, 1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 1, 2, num_register_tokens, embed_dim))
        self.trainable_cond_mask = nn.Embedding(4, embed_dim)
        self.pos_embed_rgb = nn.Parameter(torch.randn(1, 1, max_seq_len, embed_dim))
        self.pos_embed_depth = nn.Parameter(torch.randn(1, 1, max_seq_len, embed_dim))
        self.pos_embed_lidar = nn.Parameter(torch.randn(1, 1, max_seq_len, embed_dim))
        self.pos_embed_mmwave = nn.Parameter(torch.randn(1, 1, max_seq_len, embed_dim))

        self.ray_pos_proj = nn.Linear(ray_pos_dim, embed_dim, bias=proj_bias)
        self.ray_pos_norm = nn.LayerNorm(embed_dim)
        self.use_ray_pos = use_ray_pos
        self.require_ray_pos = require_ray_pos
        self.normalize_ray_pos = normalize_ray_pos
        self.use_ray_pos_gate = use_ray_pos_gate
        self.ray_pos_gate = nn.Parameter(torch.tensor(float(ray_pos_gate_init)))
        self.ray_modalities = ray_modalities if ray_modalities is not None else ["rgb", "depth"]
        self._patch_center_cache = {}

        if block_type == "Block":
            block_fn = Block
        else:
            raise NotImplementedError(f"Block type {block_type} not implemented.")

        if "single" in aa_order:
            self.single_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        qk_norm=qk_norm,
                        init_values=init_values,
                    )
                    for _ in range(depth)
                ]
            )

        if "global" in aa_order:
            self.global_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        qk_norm=qk_norm,
                        init_values=init_values,
                    )
                    for _ in range(depth)
                ]
            )

        if "cross" in aa_order:
            self.cross_blocks = nn.ModuleList(
                [
                    CABlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        qk_norm=qk_norm,
                        init_values=init_values,
                    )
                    for _ in range(depth)
                ]
            )

        self.depth = depth
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.aa_block_num = depth // aa_block_size
        self.patch_start_idx = 1 + num_register_tokens

        self.use_grad_ckpt = use_grad_ckpt

        self.anchor_map = {
            "input_rgb": 0,
            "input_depth": 1,
            "input_lidar": 2,
            "input_mmwave": 3,
        }
        self.modalities = ["rgb", "depth", "lidar", "mmwave"]

        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.joint_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        nn.init.normal_(self.trainable_cond_mask.weight, std=1e-6)

    def forward(self, features, **kwargs):
        features_rgb, features_depth, features_lidar, features_mmwave = features
        B = T = 0

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
            raise ValueError("TransformerAggregatorWithRayPos requires at least one non-empty modality.")

        camera_tokens_normal, camera_tokens_anchor = self.expand_special_tokens(self.camera_token, B, T)
        joint_tokens_normal, joint_tokens_anchor = self.expand_special_tokens(self.joint_token, B, T)
        register_tokens_normal, register_tokens_anchor = self.expand_special_tokens(self.register_token, B, T)

        anchor_key = kwargs.get("anchor_key", None)
        if isinstance(anchor_key, (list, tuple)):
            anchor_key = anchor_key[0] if len(anchor_key) > 0 else None
        anchor_idx = self.anchor_map.get(anchor_key, -1)

        features_list = []
        Ns = []
        feats = [features_rgb, features_depth, features_lidar, features_mmwave]
        ray_pos_tokens = self._build_ray_pos_tokens(
            kwargs, feats, B, T, device=camera_tokens_normal.device, dtype=camera_tokens_normal.dtype
        )

        for i, feat in enumerate(feats):
            if feat is not None:
                if ray_pos_tokens[i] is not None:
                    feat = feat + ray_pos_tokens[i]
                if i == anchor_idx:
                    feat = self.insert_special_tokens(feat, camera_tokens_anchor, joint_tokens_anchor, register_tokens_anchor)
                else:
                    feat = self.insert_special_tokens(feat, camera_tokens_normal, joint_tokens_normal, register_tokens_normal)
                features_list.append(feat)
                Ns.append(feat.shape[2])
            else:
                Ns.append(0)

        features_cat = torch.cat(features_list, dim=2)

        n_cumsum = torch.tensor([0] + Ns, device=features_cat.device).cumsum(0)
        mask_input = torch.zeros(B, T, n_cumsum[-1], dtype=torch.long, device=features_cat.device)
        for i in range(1, 4):
            if Ns[i] > 0:
                mask_input[:, :, n_cumsum[i] : n_cumsum[i + 1]] = i

        cond_mask = self.trainable_cond_mask(mask_input)
        tokens = features_cat + cond_mask
        del features_cat, cond_mask

        pos = None

        single_idx = 0
        global_idx = 0
        cross_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            block_intermediates = []
            for aa_type in self.aa_order:
                if aa_type == "single":
                    tokens, single_idx, intermediates = self._process_single_attention(tokens, Ns, single_idx, pos=pos)
                    block_intermediates.append(intermediates)
                elif aa_type == "global":
                    tokens, global_idx, intermediates = self._process_global_attention(tokens, Ns, global_idx, pos=pos)
                    block_intermediates.append(intermediates)
                elif aa_type == "cross":
                    tokens, cross_idx, intermediates = self._process_cross_attention(tokens, Ns, cross_idx, pos=pos)
                    block_intermediates.append(intermediates)

            if block_intermediates:
                max_len = max(len(b) for b in block_intermediates)
                for i in range(max_len):
                    cat_feats = [b[i] for b in block_intermediates if i < len(b)]
                    if cat_feats:
                        output_list.append(torch.cat(cat_feats, dim=-1))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return output_list

    def _build_ray_pos_tokens(self, kwargs, features, B, T, device, dtype):
        tokens = [None, None, None, None]
        if not self.use_ray_pos:
            return tokens

        for idx, modality in enumerate(self.modalities):
            if modality not in self.ray_modalities:
                continue
            feat = features[idx]
            if feat is None:
                continue
            ray_enc = self._get_ray_encoding(kwargs, modality, B, T, feat.shape[2], device, dtype)
            if ray_enc is None:
                if self.require_ray_pos:
                    raise KeyError(
                        f"Missing camera parameters for ray encoding on modality '{modality}'. "
                        f"Expected '{modality}_camera' and 'input_{modality}'."
                    )
                continue
            if self.normalize_ray_pos:
                ray_enc = self._normalize_ray_encoding(ray_enc)
            token = self.ray_pos_proj(ray_enc)
            token = self.ray_pos_norm(token)
            if self.use_ray_pos_gate:
                gate = torch.sigmoid(self.ray_pos_gate)
                tokens[idx] = token * gate
            else:
                tokens[idx] = token
        return tokens

    def _get_ray_encoding(self, kwargs, modality, B, T, num_tokens, device, dtype):
        camera_key = f"{modality}_camera"
        input_key = f"input_{modality}"
        camera = kwargs.get(camera_key)
        frames = kwargs.get(input_key)
        if camera is None or frames is None:
            return None

        frames_tensor = frames if torch.is_tensor(frames) else None
        if frames_tensor is None or frames_tensor.dim() < 4:
            return None

        height = int(frames_tensor.shape[-2])
        width = int(frames_tensor.shape[-1])

        extrinsics, intrinsics = self._stack_camera_matrices(camera, B, device, dtype)
        if extrinsics is None or intrinsics is None:
            return None

        if extrinsics.dim() == 3:
            extrinsics = extrinsics.unsqueeze(1).expand(B, T, 3, 4)
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.unsqueeze(1).expand(B, T, 3, 3)

        if extrinsics.dim() != 4 or extrinsics.shape[-2:] != (3, 4):
            return None
        if intrinsics.dim() != 4 or intrinsics.shape[-2:] != (3, 3):
            return None

        pixel_coords = self._get_patch_centers(height, width, num_tokens, device, dtype)
        rays = self._compute_world_rays(extrinsics, intrinsics, pixel_coords)
        return rays

    def _compute_world_rays(self, extrinsics, intrinsics, pixel_coords):
        B, T = extrinsics.shape[:2]
        extrinsics_flat = extrinsics.reshape(B * T, 3, 4)
        intrinsics_flat = intrinsics.reshape(B * T, 3, 3)

        R = extrinsics_flat[:, :, :3]
        t = extrinsics_flat[:, :, 3]
        R_t = R.transpose(1, 2)
        inv_K = torch.inverse(intrinsics_flat)

        cam_dirs = torch.einsum("bij,nj->bni", inv_K, pixel_coords)
        world_dirs = torch.einsum("bij,bnj->bni", R_t, cam_dirs)
        world_dirs = world_dirs / (world_dirs.norm(dim=-1, keepdim=True) + 1e-6)

        origins = -torch.einsum("bij,bj->bi", R_t, t)
        origins = origins[:, None, :].expand(-1, pixel_coords.shape[0], -1)

        rays = torch.cat([origins, world_dirs], dim=-1)
        return rays.reshape(B, T, pixel_coords.shape[0], -1)

    def _get_patch_centers(self, height, width, num_tokens, device, dtype):
        cache_key = (height, width, num_tokens, device, dtype)
        cached = self._patch_center_cache.get(cache_key)
        if cached is not None:
            return cached

        grid_h, grid_w = self._infer_patch_grid(height, width, num_tokens)
        patch_h = height / grid_h
        patch_w = width / grid_w

        y = torch.arange(grid_h, device=device, dtype=dtype) * patch_h + patch_h * 0.5
        x = torch.arange(grid_w, device=device, dtype=dtype) * patch_w + patch_w * 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        coords = torch.cat([coords, torch.ones(coords.shape[0], 1, device=device, dtype=dtype)], dim=-1)

        self._patch_center_cache[cache_key] = coords
        return coords

    def _infer_patch_grid(self, height, width, num_tokens):
        patch_guess = int(round(math.sqrt((height * width) / max(num_tokens, 1))))
        if patch_guess > 0:
            if height % patch_guess == 0 and width % patch_guess == 0:
                h_grid = height // patch_guess
                w_grid = width // patch_guess
                if h_grid * w_grid == num_tokens:
                    return h_grid, w_grid

        best_h, best_w = None, None
        target_ratio = height / max(width, 1)
        max_h = int(math.sqrt(num_tokens))
        for h in range(1, max_h + 1):
            if num_tokens % h != 0:
                continue
            w = num_tokens // h
            ratio = h / w
            if best_h is None or abs(ratio - target_ratio) < abs((best_h / best_w) - target_ratio):
                best_h, best_w = h, w

        if best_h is None:
            best_h = int(round(math.sqrt(num_tokens)))
            best_w = max(1, num_tokens // max(best_h, 1))

        return best_h, best_w

    def _stack_camera_matrices(self, camera, B, device, dtype):
        if isinstance(camera, list):
            if len(camera) == 0:
                return None, None
            if len(camera) == 1 and B > 1:
                camera = camera * B
            if len(camera) != B:
                return None, None
            extrinsics = torch.stack(
                [self._as_tensor(cam.get("extrinsic"), device, dtype) for cam in camera], dim=0
            )
            intrinsics = torch.stack(
                [self._as_tensor(cam.get("intrinsic"), device, dtype) for cam in camera], dim=0
            )
        else:
            extrinsic = self._as_tensor(getattr(camera, "get", lambda x: None)("extrinsic"), device, dtype)
            intrinsic = self._as_tensor(getattr(camera, "get", lambda x: None)("intrinsic"), device, dtype)
            if extrinsic is None or intrinsic is None:
                return None, None
            extrinsics = extrinsic.unsqueeze(0).expand(B, -1, -1)
            intrinsics = intrinsic.unsqueeze(0).expand(B, -1, -1)

        if extrinsics.dim() not in (3, 4):
            return None, None
        if intrinsics.dim() not in (3, 4):
            return None, None
        return extrinsics, intrinsics

    def _as_tensor(self, value, device, dtype):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _normalize_ray_encoding(self, ray_enc):
        mean = ray_enc.mean(dim=(0, 1, 2), keepdim=True)
        std = ray_enc.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-6)
        return (ray_enc - mean) / std

    def _process_single_attention(self, tokens, Ns, idx, pos=None):
        B, T, N_total, C = tokens.shape
        n_cumsum = torch.tensor([0] + list(Ns), device=tokens.device).cumsum(0)
        intermediates = []
        tokens_base = tokens

        for _ in range(self.aa_block_size):
            updated_slices = []
            for i in range(4):
                start, end = n_cumsum[i].item(), n_cumsum[i + 1].item()
                if end > start:
                    tokens_slice = tokens_base[:, :, start:end, :].reshape(B, T * (end - start), C)
                    if self.use_grad_ckpt and self.training:
                        tokens_slice = checkpoint(self.single_blocks[idx], tokens_slice, pos, use_reentrant=False)
                    else:
                        tokens_slice = self.single_blocks[idx](tokens_slice, pos=pos)
                    updated_slices.append((i, tokens_slice.reshape(B, T, end - start, C)))

            new_tokens = tokens_base.clone()
            for i, slice_tensor in updated_slices:
                start, end = n_cumsum[i].item(), n_cumsum[i + 1].item()
                new_tokens[:, :, start:end, :] = slice_tensor

            tokens_base = new_tokens
            idx += 1
            intermediates.append(self.extract_output_tokens(tokens_base, Ns))

        return tokens_base, idx, intermediates

    def _process_global_attention(self, tokens, Ns, idx, pos=None):
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
        tokens = tokens.expand(B, T, -1, -1, -1)
        tokens_normal, tokens_anchor = tokens[:, :, 0], tokens[:, :, 1]
        return tokens_normal, tokens_anchor

    def insert_special_tokens(self, features, camera_tokens, joint_tokens, register_tokens):
        if features is not None:
            features = torch.cat([camera_tokens, joint_tokens, register_tokens, features], dim=2)
        return features

    def extract_output_tokens(self, tokens, Ns):
        B, T, _, C = tokens.shape

        token_slices = []
        cumsum = 0
        for n in Ns:
            if n > 0:
                token_slices.append(tokens[:, :, cumsum : cumsum + 2, :].contiguous())
            cumsum += n

        if len(token_slices) == 0:
            return tokens.new_zeros(B, 0, T, 2, C)

        output = torch.stack(token_slices, dim=1)
        return output
