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


class TransformerAggregatorV2GlobalSMPL(nn.Module):
    def __init__(
        self,
        input_dims: List[int] = [512, 512, 512, 512],
        embed_dim: int = 512,
        num_register_tokens: int = 4,
        num_smpl_tokens: int = 1,
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

        self.camera_token = nn.Parameter(torch.randn(1, 1, 1, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 1, 1, num_register_tokens, embed_dim))
        self.smpl_token = nn.Parameter(torch.randn(1, 1, num_smpl_tokens, embed_dim))
        self.joint_token = nn.Parameter(torch.randn(1, 1, self.num_joints, embed_dim))
        self.num_register_tokens = num_register_tokens
        self.num_smpl_tokens = num_smpl_tokens

        self.modality_embed = nn.Parameter(torch.randn(1, 1, 4, embed_dim))

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

        if "single" in aa_order:
            self.single_blocks = nn.ModuleList([block_cls(**block_params) for _ in range(depth)])
        if "cross_joint" in aa_order:
            self.cross_joint_blocks = nn.ModuleList([CABlock(**block_params) for _ in range(depth)])
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

        B, T = 0, 0
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

        camera_tokens = self.camera_token.expand(B, T, -1, -1, -1)[:, :, 0]
        register_tokens = self.register_token.expand(B, T, -1, -1, -1)[:, :, 0]
        smpl_tokens = self.smpl_token.expand(B, T, -1, -1)
        joint_tokens = self.joint_token.expand(B, T, -1, -1)

        # Concatenate special tokens with patch tokens
        modality_tokens = []
        Ns = []
        for i, feat in enumerate([features_rgb, features_depth, features_lidar, features_mmwave]):
            if feat is not None:
                feat = self._insert_special_tokens(feat, camera_tokens, register_tokens)
                feat = feat + self.modality_embed[:, :, i, :].unsqueeze(2)
                modality_tokens.append(feat)
                Ns.append(feat.shape[2])
            else:
                modality_tokens.append(None)
                Ns.append(0)

        single_idx = 0
        cross_idx = 0
        gcn_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for aa_type in self.aa_order:
                match aa_type:
                    case "single":
                        modality_tokens, single_idx, intermediates = self._process_single_attention(
                            modality_tokens, smpl_tokens, joint_tokens, Ns, single_idx, pos=None
                        )
                    case "cross_joint":
                        smpl_tokens, joint_tokens, cross_idx, intermediates = self._process_cross_joint_attention(
                            smpl_tokens, joint_tokens, modality_tokens, Ns, cross_idx, pos=None
                        )
                    case "gcn":
                        joint_tokens, gcn_idx, intermediates = self._process_gcn(
                            smpl_tokens, joint_tokens, modality_tokens, Ns, gcn_idx
                        )
                    case _:
                        raise ValueError(f"Unknown attention type: {aa_type}")
            output_list.append(intermediates[-1])

        return output_list

    def _process_single_attention(self, modality_tokens, smpl_tokens, joint_tokens, Ns, idx, pos=None):
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
            intermediates.append(self._extract_output_tokens(tokens_base, smpl_tokens, joint_tokens, Ns))

        return tokens_base, idx, intermediates

    def _process_cross_joint_attention(self, smpl_tokens, joint_tokens, modality_tokens, Ns, idx, pos=None):
        intermediates = []
        smpl_base = smpl_tokens
        joint_base = joint_tokens

        for _ in range(self.aa_block_size):
            pose_base = torch.cat([smpl_base, joint_base], dim=2)
            queries_base = pose_base.reshape(
                pose_base.shape[0],
                pose_base.shape[1] * (self.num_smpl_tokens + self.num_joints),
                pose_base.shape[3],
            )
            summed_queries = None
            num_modalities = 0
            for tokens in modality_tokens:
                if tokens is None:
                    continue
                B, T, N, C = tokens.shape
                context = tokens.reshape(B, T * N, C)

                if self.use_grad_ckpt and self.training:
                    queries = checkpoint(
                        self.cross_joint_blocks[idx], queries_base, context, pos, use_reentrant=False
                    )
                else:
                    queries = self.cross_joint_blocks[idx](queries_base, context=context, pos=pos)

                if summed_queries is None:
                    summed_queries = queries
                else:
                    summed_queries = summed_queries + queries
                num_modalities += 1

            if summed_queries is None:
                summed_queries = queries_base
            else:
                summed_queries = summed_queries / float(num_modalities)

            pose_base = summed_queries.reshape(
                pose_base.shape[0],
                pose_base.shape[1],
                self.num_smpl_tokens + self.num_joints,
                pose_base.shape[3],
            )
            smpl_base = pose_base[:, :, :self.num_smpl_tokens, :]
            joint_base = pose_base[:, :, self.num_smpl_tokens:, :]

            idx += 1
            intermediates.append(self._extract_output_tokens(modality_tokens, smpl_base, joint_base, Ns))

        return smpl_base, joint_base, idx, intermediates

    def _process_gcn(self, smpl_tokens, joint_tokens, modality_tokens, Ns, idx):
        intermediates = []
        joint_base = joint_tokens

        for _ in range(self.aa_block_size):
            joint_gcn = rearrange(joint_base, "b t v c -> b c t v")

            if self.use_grad_ckpt and self.training:
                joint_gcn = checkpoint(self.gcn_blocks[idx], joint_gcn, use_reentrant=False)
            else:
                joint_gcn = self.gcn_blocks[idx](joint_gcn)

            joint_base = rearrange(joint_gcn, "b c t v -> b t v c")
            idx += 1
            intermediates.append(self._extract_output_tokens(modality_tokens, smpl_tokens, joint_base, Ns))

        return joint_base, idx, intermediates

    def _insert_special_tokens(self, features: Tensor, camera_tokens: Tensor, register_tokens: Tensor):
        if features is not None:
            features = torch.cat([camera_tokens, register_tokens, features], dim=2)
        return features

    def _extract_output_tokens(self, modality_tokens, smpl_tokens, joint_tokens, Ns):
        B, T, _, C = joint_tokens.shape
        camera_slices = []

        for tokens in modality_tokens:
            if tokens is None:
                continue
            camera_tokens = tokens[:, :, :1, :].contiguous()
            camera_slices.append(camera_tokens)

        if len(camera_slices) == 0:
            return joint_tokens.new_zeros(B, T, self.num_smpl_tokens + self.num_joints, C)

        camera_tokens = torch.cat(camera_slices, dim=2)
        return torch.cat([camera_tokens, smpl_tokens, joint_tokens], dim=2)
