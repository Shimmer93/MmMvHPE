import torch
import torch.nn as nn
from typing import List, Optional
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .trans_aggregator_v4 import TransformerAggregatorV4
from .layers.block import CABlock
from .layers.gcn import TCN_GCN_unit as GCNBlock
from misc.skeleton import (
    get_adjacency_matrix,
    COCOSkeleton,
    SimpleCOCOSkeleton,
    H36MSkeleton,
    SMPLSkeleton,
)


class TransformerAggregatorV5(TransformerAggregatorV4):
    """V4 aggregator with JSON-skeleton-guided token update.

    Pipeline addition vs V4:
    1. Read JSON-loaded skeleton keypoints (e.g., gt_keypoints_2d_rgb).
    2. Encode these keypoints with a dedicated GCN using the JSON skeleton graph.
    3. Use cross-attention (learnable tokens as queries, JSON skeleton as context)
       to update learnable special tokens before standard V4 processing.
    """

    def __init__(
        self,
        input_dims: List[int] = [512, 512, 512, 512],
        embed_dim: int = 512,
        num_register_tokens: int = 4,
        num_smpl_tokens: int = 1,
        max_modalities: int = 4,
        aa_order: List[str] = ["single", "cross_joint", "gcn"],
        aa_block_size: int = 1,
        depth: int = 24,
        block_type: str = "Block",
        skeleton_type: str = "smpl",
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 0.01,
        use_grad_ckpt: bool = False,
        json_skeleton_key: str = "gt_keypoints_2d_rgb",
        json_skeleton_type: str = "coco",
        json_skeleton_in_dim: int = 2,
        json_skeleton_gcn_depth: int = 2,
        json_skeleton_attn_depth: int = 1,
    ):
        super().__init__(
            input_dims=input_dims,
            embed_dim=embed_dim,
            num_register_tokens=num_register_tokens,
            num_smpl_tokens=num_smpl_tokens,
            max_modalities=max_modalities,
            aa_order=aa_order,
            aa_block_size=aa_block_size,
            depth=depth,
            block_type=block_type,
            skeleton_type=skeleton_type,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            use_grad_ckpt=use_grad_ckpt,
        )

        self.json_skeleton_key = json_skeleton_key
        self.json_skeleton_in_dim = int(max(1, json_skeleton_in_dim))
        self.json_skeleton_gcn_depth = int(max(1, json_skeleton_gcn_depth))
        self.json_skeleton_attn_depth = int(max(1, json_skeleton_attn_depth))

        json_skeleton = self._build_json_skeleton(json_skeleton_type)
        self.json_num_joints = json_skeleton.num_joints
        A_json = get_adjacency_matrix(json_skeleton.bones, json_skeleton.num_joints)

        self.json_input_proj = nn.Sequential(
            nn.Linear(self.json_skeleton_in_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim),
        )
        self.json_gcn_blocks = nn.ModuleList(
            [GCNBlock(embed_dim, embed_dim, A_json, adaptive=True) for _ in range(self.json_skeleton_gcn_depth)]
        )

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
        self.json_token_attn_blocks = nn.ModuleList(
            [CABlock(**block_params) for _ in range(self.json_skeleton_attn_depth)]
        )

        self.json_input_proj.apply(self._init_weights)
        self.json_gcn_blocks.apply(self._init_weights)
        self.json_token_attn_blocks.apply(self._init_weights)

    @staticmethod
    def _build_json_skeleton(json_skeleton_type: str):
        skeleton_type = json_skeleton_type.lower()
        if skeleton_type == "coco":
            return COCOSkeleton()
        if skeleton_type == "simplecoco":
            return SimpleCOCOSkeleton()
        if skeleton_type in {"h36m", "mmfi"}:
            return H36MSkeleton()
        if skeleton_type == "smpl":
            return SMPLSkeleton()
        raise ValueError(
            f"Unknown json_skeleton_type: {json_skeleton_type}. "
            "Supported: coco, simplecoco, h36m/mmfi, smpl."
        )

    @staticmethod
    def _coerce_tensor(value) -> Optional[Tensor]:
        if value is None:
            return None
        if isinstance(value, Tensor):
            return value
        if isinstance(value, (list, tuple)):
            valid = [v for v in value if v is not None]
            if not valid:
                return None
            if all(isinstance(v, Tensor) for v in valid):
                if any(v.shape != valid[0].shape for v in valid):
                    return None
                return torch.stack(valid, dim=0)
            try:
                return torch.as_tensor(valid, dtype=torch.float32)
            except Exception:
                return None
        try:
            return torch.as_tensor(value, dtype=torch.float32)
        except Exception:
            return None

    def _align_json_skeleton(self, skeleton: Tensor, target_t: int) -> Optional[Tensor]:
        if skeleton is None:
            return None
        if skeleton.dim() == 3:
            skeleton = skeleton.unsqueeze(0)
        if skeleton.dim() != 4:
            return None

        if skeleton.shape[-1] > self.json_skeleton_in_dim:
            skeleton = skeleton[..., : self.json_skeleton_in_dim]
        elif skeleton.shape[-1] < self.json_skeleton_in_dim:
            pad_dim = self.json_skeleton_in_dim - skeleton.shape[-1]
            pad = torch.zeros(*skeleton.shape[:-1], pad_dim, device=skeleton.device, dtype=skeleton.dtype)
            skeleton = torch.cat([skeleton, pad], dim=-1)

        joints = skeleton.shape[2]
        if joints > self.json_num_joints:
            skeleton = skeleton[:, :, : self.json_num_joints, :]
        elif joints < self.json_num_joints:
            pad_j = self.json_num_joints - joints
            pad = torch.zeros(
                skeleton.shape[0], skeleton.shape[1], pad_j, skeleton.shape[3],
                device=skeleton.device, dtype=skeleton.dtype
            )
            skeleton = torch.cat([skeleton, pad], dim=2)

        if skeleton.shape[1] < target_t:
            repeat = target_t - skeleton.shape[1]
            tail = skeleton[:, -1:, :, :].expand(-1, repeat, -1, -1)
            skeleton = torch.cat([skeleton, tail], dim=1)
        elif skeleton.shape[1] > target_t:
            skeleton = skeleton[:, :target_t]

        return skeleton

    def _encode_json_skeleton(self, json_skeleton: Tensor) -> Tensor:
        x = self.json_input_proj(json_skeleton)
        x = rearrange(x, "b t v c -> b c t v")
        for gcn in self.json_gcn_blocks:
            if self.use_grad_ckpt and self.training:
                x = checkpoint(gcn, x, use_reentrant=False)
            else:
                x = gcn(x)
        x = rearrange(x, "b c t v -> b t v c")
        return x

    def _update_special_tokens_with_json_skeleton(
        self,
        camera_tokens: Tensor,
        register_tokens: Tensor,
        smpl_tokens: Tensor,
        joint_tokens: Tensor,
        json_skeleton_raw,
    ):
        json_skeleton = self._coerce_tensor(json_skeleton_raw)
        if json_skeleton is None:
            return camera_tokens, register_tokens, smpl_tokens, joint_tokens

        B, T, M, _, _ = joint_tokens.shape
        json_skeleton = json_skeleton.to(device=joint_tokens.device, dtype=joint_tokens.dtype)
        json_skeleton = self._align_json_skeleton(json_skeleton, target_t=T)
        if json_skeleton is None:
            return camera_tokens, register_tokens, smpl_tokens, joint_tokens
        if json_skeleton.shape[0] != B:
            if json_skeleton.shape[0] == 1 and B > 1:
                json_skeleton = json_skeleton.expand(B, -1, -1, -1)
            else:
                return camera_tokens, register_tokens, smpl_tokens, joint_tokens

        json_skeleton = torch.nan_to_num(json_skeleton, nan=0.0, posinf=0.0, neginf=0.0)
        valid = json_skeleton.abs().sum(dim=(1, 2, 3)) > 1e-6
        if valid.sum().item() == 0:
            return camera_tokens, register_tokens, smpl_tokens, joint_tokens

        context = self._encode_json_skeleton(json_skeleton)
        context = rearrange(context, "b t v c -> b (t v) c")

        special_tokens = torch.cat([camera_tokens, register_tokens, smpl_tokens, joint_tokens], dim=3)
        num_special = special_tokens.shape[3]
        queries = rearrange(special_tokens, "b t m n c -> b (t m n) c")

        for attn in self.json_token_attn_blocks:
            if self.use_grad_ckpt and self.training:
                queries = checkpoint(attn, queries, context, None, use_reentrant=False)
            else:
                queries = attn(queries, context=context, pos=None)

        updated = rearrange(queries, "b (t m n) c -> b t m n c", t=T, m=M, n=num_special)
        valid_mask = valid.view(B, 1, 1, 1, 1)
        special_tokens = torch.where(valid_mask, updated, special_tokens)

        c_end = 1
        r_end = c_end + self.num_register_tokens
        s_end = r_end + self.num_smpl_tokens
        j_end = s_end + self.num_joints
        camera_tokens = special_tokens[:, :, :, :c_end, :]
        register_tokens = special_tokens[:, :, :, c_end:r_end, :]
        smpl_tokens = special_tokens[:, :, :, r_end:s_end, :]
        joint_tokens = special_tokens[:, :, :, s_end:j_end, :]
        return camera_tokens, register_tokens, smpl_tokens, joint_tokens

    def forward(self, features, **kwargs):
        features_rgb, features_depth, features_lidar, features_mmwave = features

        B, T, M = 0, 0, 0
        if features_rgb is not None:
            features_rgb = self.proj_rgb(features_rgb)
            B, T, _, _ = features_rgb.shape
            M += 1
        if features_depth is not None:
            features_depth = self.proj_depth(features_depth)
            B, T, _, _ = features_depth.shape
            M += 1
        if features_lidar is not None:
            features_lidar = self.proj_lidar(features_lidar)
            B, T, _, _ = features_lidar.shape
            M += 1
        if features_mmwave is not None:
            features_mmwave = self.proj_mmwave(features_mmwave)
            B, T, _, _ = features_mmwave.shape
            M += 1

        if M == 0:
            raise ValueError("At least one modality must be provided.")

        camera_tokens = self.camera_token.expand(B, T, M, -1, -1)
        register_tokens = self.register_token.expand(B, T, M, -1, -1)
        smpl_tokens = self.smpl_token.expand(B, T, M, -1, -1)
        joint_tokens = self.joint_token.expand(B, T, M, -1, -1)

        camera_tokens, register_tokens, smpl_tokens, joint_tokens = (
            self._update_special_tokens_with_json_skeleton(
                camera_tokens,
                register_tokens,
                smpl_tokens,
                joint_tokens,
                kwargs.get(self.json_skeleton_key, None),
            )
        )

        special_tokens = torch.cat([camera_tokens, register_tokens, smpl_tokens, joint_tokens], dim=3)
        modality_tokens = []
        Ns = []
        j = 0
        for i, feat in enumerate([features_rgb, features_depth, features_lidar, features_mmwave]):
            if feat is not None:
                feat = torch.cat([feat, special_tokens[:, :, j, :, :]], dim=2)
                feat = self._insert_special_tokens(feat, special_tokens[:, :, j, :, :])
                feat = feat + self.modality_embed[:, :, i, :].unsqueeze(2)
                modality_tokens.append(feat)
                Ns.append(feat.shape[2])
                j += 1
            else:
                modality_tokens.append(None)
                Ns.append(0)

        single_idx = 0
        cross_idx = 0
        cross_modality_idx = 0
        joint_to_camera_idx = 0
        gcn_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for aa_type in self.aa_order:
                match aa_type:
                    case "single":
                        modality_tokens, single_idx, intermediates = self._process_single_attention(
                            modality_tokens, Ns, single_idx, pos=None
                        )
                    case "cross_joint":
                        modality_tokens, cross_idx, intermediates = self._process_cross_joint_attention(
                            modality_tokens, Ns, cross_idx, pos=None
                        )
                    case "cross_modality":
                        modality_tokens, cross_modality_idx, intermediates = self._process_cross_modality_attention(
                            modality_tokens, Ns, cross_modality_idx, pos=None
                        )
                    case "joint_to_camera":
                        modality_tokens, joint_to_camera_idx, intermediates = self._process_joint_to_camera_attention(
                            modality_tokens, Ns, joint_to_camera_idx, pos=None
                        )
                    case "gcn":
                        modality_tokens, gcn_idx, intermediates = self._process_gcn(
                            modality_tokens, Ns, gcn_idx
                        )
                    case _:
                        raise ValueError(f"Unknown attention type: {aa_type}")
            output_list.extend(intermediates)

        return output_list
