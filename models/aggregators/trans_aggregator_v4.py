import torch
import torch.nn as nn
from typing import List, Optional
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from models.video_encoders.layers.block import Block
from .layers.block import CABlock
from .layers.gcn import TCN_GCN_unit as GCNBlock
from misc.skeleton import get_adjacency_matrix, H36MSkeleton, SMPLSkeleton

class TransformerAggregatorV4(nn.Module):
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
    ):
        super().__init__()

        if depth % aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        match skeleton_type:
            case "smpl":
                skeleton = SMPLSkeleton()
            case _:
                raise ValueError(f"Unknown skeleton type: {skeleton_type}")

        self.num_joints = skeleton.num_joints

        rgb_dim, depth_dim, lidar_dim, mmwave_dim = input_dims
        self.proj_rgb = nn.Sequential(
            nn.Linear(rgb_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim),
        )
        self.proj_depth = nn.Sequential(
            nn.Linear(depth_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim),
        )
        self.proj_lidar = nn.Sequential(
            nn.Linear(lidar_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim),
        )
        self.proj_mmwave = nn.Sequential(
            nn.Linear(mmwave_dim, embed_dim, bias=proj_bias),
            nn.LayerNorm(embed_dim),
        )

        self.camera_token = nn.Parameter(torch.randn(1, 1, 1, 1, embed_dim)) # B T M 1 D
        self.register_token = nn.Parameter(torch.randn(1, 1, 1, num_register_tokens, embed_dim)) # B T M R D
        self.smpl_token = nn.Parameter(torch.randn(1, 1, 1, num_smpl_tokens, embed_dim)) # B T M S D
        self.joint_token = nn.Parameter(torch.randn(1, 1, 1, self.num_joints, embed_dim)) # B T M J D
        self.num_register_tokens = num_register_tokens
        self.num_smpl_tokens = num_smpl_tokens
        self.max_modalities = max_modalities

        self.modality_embed = nn.Parameter(torch.randn(1, 1, 4, embed_dim))

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

        if "single" in aa_order:
            self.single_blocks = nn.ModuleList([Block(**block_params) for _ in range(depth)])
        if "cross_modality" in aa_order:
            self.cross_modality_blocks = nn.ModuleList([Block(**block_params) for _ in range(depth)])
        if "cross_joint" in aa_order:
            self.cross_joint_blocks = nn.ModuleList([CABlock(**block_params) for _ in range(depth)])
        if "joint_to_camera" in aa_order:
            self.joint_to_camera_blocks = nn.ModuleList([CABlock(**block_params) for _ in range(depth)])
        if "gcn" in aa_order:
            A = get_adjacency_matrix(skeleton.bones, skeleton.num_joints)
            self.gcn_blocks = nn.ModuleList(
                [GCNBlock(embed_dim, embed_dim, A, adaptive=True) for _ in range(depth)]
            )

        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.depth = depth
        self.aa_block_num = depth // aa_block_size

        self.use_grad_ckpt = use_grad_ckpt

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
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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

        # Concatenate special tokens with patch tokens
        special_tokens = torch.cat([camera_tokens, register_tokens, smpl_tokens, joint_tokens], dim=3) # B T M S D
        modality_tokens = []
        Ns = []
        j = 0
        for i, feat in enumerate([features_rgb, features_depth, features_lidar, features_mmwave]):
            if feat is not None:
                feat = torch.cat([feat, special_tokens[:, :, j, :, :]], dim=2)  # B T N+S D
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

    def _process_single_attention(self, modality_tokens, Ns, idx, pos=None):
        intermediates = []
        tokens_base = modality_tokens

        for _ in range(self.aa_block_size):
            updated = list(tokens_base)
            for i, tokens in enumerate(tokens_base):
                if tokens is None:
                    continue
                B, T, N, C = tokens.shape
                tokens_reshaped = tokens.reshape(B, T * N, C)

                if self.use_grad_ckpt and self.training:
                    tokens_reshaped = checkpoint(self.single_blocks[idx], tokens_reshaped, pos, use_reentrant=False)
                else:
                    tokens_reshaped = self.single_blocks[idx](tokens_reshaped, pos=pos)

                updated[i] = tokens_reshaped.reshape(B, T, N, C)

            tokens_base = updated
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base))

        return tokens_base, idx, intermediates

    def _process_cross_joint_attention(self, modality_tokens, Ns, idx, pos=None):
        intermediates = []
        tokens_base = modality_tokens
        num_camera = 1
        num_register = self.num_register_tokens
        num_pose = self.num_smpl_tokens + self.num_joints
        pose_start = num_camera + num_register
        pose_end = pose_start + num_pose

        for _ in range(self.aa_block_size):
            updated = list(tokens_base)
            for i, tokens in enumerate(tokens_base):
                if tokens is None:
                    continue
                B, T, N, C = tokens.shape
                pose_tokens = tokens[:, :, pose_start:pose_end, :]
                queries = pose_tokens.reshape(B, T * num_pose, C)
                context = tokens.reshape(B, T * N, C)

                if self.use_grad_ckpt and self.training:
                    queries = checkpoint(
                        self.cross_joint_blocks[idx], queries, context, pos, use_reentrant=False
                    )
                else:
                    queries = self.cross_joint_blocks[idx](queries, context=context, pos=pos)

                pose_tokens = queries.reshape(B, T, num_pose, C)
                updated[i] = torch.cat(
                    [tokens[:, :, :pose_start, :], pose_tokens, tokens[:, :, pose_end:, :]], dim=2
                )

            tokens_base = updated
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base))

        return tokens_base, idx, intermediates

    def _process_cross_modality_attention(self, modality_tokens, Ns, idx, pos=None):
        intermediates = []
        tokens_base = modality_tokens
        num_special = 1 + self.num_register_tokens + self.num_smpl_tokens + self.num_joints

        for _ in range(self.aa_block_size):
            patch_slices = []
            patch_sizes = []
            for tokens in tokens_base:
                if tokens is None:
                    patch_sizes.append(0)
                    continue
                patch = tokens[:, :, num_special:, :]
                patch_slices.append(patch)
                patch_sizes.append(patch.shape[2])

            if patch_slices:
                merged_patches = torch.cat(patch_slices, dim=2)
                B, T, N, C = merged_patches.shape
                merged_patches = merged_patches.reshape(B, T * N, C)

                if self.use_grad_ckpt and self.training:
                    merged_patches = checkpoint(
                        self.cross_modality_blocks[idx], merged_patches, pos, use_reentrant=False
                    )
                else:
                    merged_patches = self.cross_modality_blocks[idx](merged_patches, pos=pos)

                merged_patches = merged_patches.reshape(B, T, N, C)

                updated = []
                cursor = 0
                for tokens, size in zip(tokens_base, patch_sizes):
                    if tokens is None:
                        updated.append(None)
                        continue
                    if size == 0:
                        updated.append(tokens)
                        continue
                    patch = merged_patches[:, :, cursor : cursor + size, :]
                    cursor += size
                    updated.append(torch.cat([tokens[:, :, :num_special, :], patch], dim=2))
                tokens_base = updated

            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base))

        return tokens_base, idx, intermediates

    def _process_joint_to_camera_attention(self, modality_tokens, Ns, idx, pos=None):
        intermediates = []
        tokens_base = modality_tokens
        num_camera = 1
        num_register = self.num_register_tokens
        joint_start = num_camera + num_register + self.num_smpl_tokens
        joint_end = joint_start + self.num_joints

        for _ in range(self.aa_block_size):
            updated = list(tokens_base)
            for i, tokens in enumerate(tokens_base):
                if tokens is None:
                    continue
                B, T, _, C = tokens.shape
                camera_tokens = tokens[:, :, :1, :].reshape(B, T, C)
                joint_tokens = tokens[:, :, joint_start:joint_end, :].reshape(B, T * self.num_joints, C)

                if self.use_grad_ckpt and self.training:
                    updated_camera = checkpoint(
                        self.joint_to_camera_blocks[idx], camera_tokens, joint_tokens, pos, use_reentrant=False
                    )
                else:
                    updated_camera = self.joint_to_camera_blocks[idx](
                        camera_tokens, context=joint_tokens, pos=pos
                    )

                updated_camera = updated_camera.reshape(B, T, 1, C)
                updated[i] = torch.cat([updated_camera, tokens[:, :, 1:, :]], dim=2)

            tokens_base = updated
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base))

        return tokens_base, idx, intermediates

    def _process_gcn(self, modality_tokens, Ns, idx):
        intermediates = []
        tokens_base = modality_tokens
        num_camera = 1
        num_register = self.num_register_tokens
        joint_start = num_camera + num_register + self.num_smpl_tokens
        joint_end = joint_start + self.num_joints

        for _ in range(self.aa_block_size):
            joint_slices = []
            active_indices = []
            for i, tokens in enumerate(tokens_base):
                if tokens is None:
                    continue
                joint_slices.append(tokens[:, :, joint_start:joint_end, :])
                active_indices.append(i)

            if joint_slices:
                joint_tokens = torch.stack(joint_slices, dim=0).mean(dim=0)
                joint_gcn = rearrange(joint_tokens, "b t v c -> b c t v")

                if self.use_grad_ckpt and self.training:
                    joint_gcn = checkpoint(self.gcn_blocks[idx], joint_gcn, use_reentrant=False)
                else:
                    joint_gcn = self.gcn_blocks[idx](joint_gcn)

                joint_tokens = rearrange(joint_gcn, "b c t v -> b t v c")
                updated = list(tokens_base)
                for i in active_indices:
                    tokens = tokens_base[i]
                    updated[i] = torch.cat(
                        [tokens[:, :, :joint_start, :], joint_tokens, tokens[:, :, joint_end:, :]], dim=2
                    )
                tokens_base = updated
            idx += 1
            intermediates.append(self._extract_output_tokens(tokens_base))

        return tokens_base, idx, intermediates

    def _insert_special_tokens(self, features: Tensor, special_tokens: Tensor):
        if features is None:
            return features

        special_len = special_tokens.shape[2]
        if features.shape[2] >= special_len:
            patch_tokens = features[:, :, : features.shape[2] - special_len, :]
        else:
            patch_tokens = features

        return torch.cat([special_tokens, patch_tokens], dim=2)

    def _extract_output_tokens(self, modality_tokens):
        token_slices = []
        for tokens in modality_tokens:
            if tokens is None:
                continue
            special_tokens = tokens[:, :, : 1 + self.num_register_tokens + self.num_smpl_tokens + self.num_joints, :]
            token_slices.append(special_tokens)

        if len(token_slices) == 0:
            return torch.zeros(
                0, 0, 0,
                1 + self.num_register_tokens + self.num_smpl_tokens + self.num_joints,
                self.joint_token.shape[-1],
                device=self.joint_token.device,
            )

        return torch.stack(token_slices, dim=2)
