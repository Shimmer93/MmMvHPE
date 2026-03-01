import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionFormerModalityAggregator(nn.Module):
    def __init__(
        self,
        num_joints: int = 24,
        joint_embed_dim: int = 128,
        pose_embed_dim: int = 256,
        efc_hidden_dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_blocks: int = 2,
        seq_len: int = 1,
        max_modalities: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.max_modalities = max_modalities
        self.num_blocks = num_blocks

        self.rgb_embed = nn.Linear(2, joint_embed_dim)
        self.depth_embed = nn.Linear(3, joint_embed_dim)
        # EFC baseline: 3-layer FCN over flattened joint embeddings (J * Cj -> Cp).
        self.efc = nn.Sequential(
            nn.Linear(num_joints * joint_embed_dim, efc_hidden_dim),
            nn.ReLU(),
            nn.Linear(efc_hidden_dim, efc_hidden_dim),
            nn.ReLU(),
            nn.Linear(efc_hidden_dim, pose_embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pose_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(pose_embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=pose_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(pose_embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        max_tokens = max_modalities * seq_len
        self.pos_enc = nn.Parameter(torch.zeros(1, max_tokens, pose_embed_dim))
        self.pos_dec = nn.Parameter(torch.zeros(1, seq_len, pose_embed_dim))
        self.pre_ln = nn.LayerNorm(pose_embed_dim)
        self.dec_ln = nn.LayerNorm(pose_embed_dim)

        self.temporal_conv = nn.Conv1d(
            in_channels=pose_embed_dim,
            out_channels=pose_embed_dim,
            kernel_size=seq_len,
            stride=1,
            padding=0,
            bias=True,
        )
        self.head = nn.Sequential(
            nn.Linear(pose_embed_dim, pose_embed_dim),
            nn.ReLU(),
            nn.Linear(pose_embed_dim, num_joints * 3),
        )

    def _embed_pose(self, pose, is_rgb: bool):
        if pose is None:
            return None
        if pose.dim() == 4:
            pose = pose.unsqueeze(1)  # B, 1, T, J, D
        if pose.dim() != 5:
            raise ValueError(f"Expected pose tensor with shape (B, T, J, D) or (B, V, T, J, D), got {pose.shape}")
        B, V, T, J, D = pose.shape  # B=batch, V=views/modalities, T=frames, J=joints, D=coord dim
        if J != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {J}")
        embed = self.rgb_embed if is_rgb else self.depth_embed
        pose = embed(pose)  # B, V, T, J, joint_embed_dim
        # Flatten joints and apply EFC to aggregate joint relationships per (V, T).
        pose = pose.reshape(B, V, T, J * pose.shape[-1])  # B, V, T, (J*Cj)
        pose = self.efc(pose)  # B, V, T, pose_embed_dim
        return pose  # B, V, T, C

    def forward(self, features, **kwargs):
        features_rgb, features_depth, features_lidar, features_mmwave = features
        # features_* expected as (B, T, J, D) or (B, V, T, J, D)
        pose_rgb = self._embed_pose(features_rgb, is_rgb=True) if features_rgb is not None else None
        pose_depth = self._embed_pose(features_depth, is_rgb=False) if features_depth is not None else None

        poses = []
        if pose_rgb is not None:
            poses.append(pose_rgb)
        if pose_depth is not None:
            poses.append(pose_depth)

        if not poses:
            raise ValueError("FusionFormerModalityAggregator requires at least one pose modality.")

        pose_stack = torch.cat(poses, dim=1)  # B, M, T, C (M = total modalities/views)
        B, M, T, C = pose_stack.shape
        tokens = pose_stack.reshape(B, M * T, C)  # B, (M*T), C

        if self.pos_enc.shape[1] != M * T:
            pos_enc = self.pos_enc.transpose(1, 2)
            pos_enc = F.interpolate(pos_enc, size=M * T, mode="linear", align_corners=False)
            pos_enc = pos_enc.transpose(1, 2)
        else:
            pos_enc = self.pos_enc

        tokens = self.pre_ln(tokens + pos_enc[:, : M * T])  # B, (M*T), C

        for _ in range(self.num_blocks):
            global_feat = self.encoder(tokens)  # B, (M*T), C
            fused = []
            for m in range(M):
                tgt = pose_stack[:, m]  # B, T, C
                if self.pos_dec.shape[1] != T:
                    pos_dec = self.pos_dec.transpose(1, 2)
                    pos_dec = F.interpolate(pos_dec, size=T, mode="linear", align_corners=False)
                    pos_dec = pos_dec.transpose(1, 2)
                else:
                    pos_dec = self.pos_dec
                tgt = self.dec_ln(tgt + pos_dec[:, :T])  # B, T, C
                dec_out = self.decoder(tgt, global_feat)  # B, T, C
                fused.append(dec_out)
            pose_stack = torch.stack(fused, dim=1)  # B, M, T, C
            tokens = pose_stack.reshape(B, M * T, C)  # B, (M*T), C

        pose_stack = pose_stack.reshape(B * M, T, C).transpose(1, 2)  # (B*M), C, T
        kernel = self.temporal_conv.kernel_size[0]
        if T < kernel:
            agg = pose_stack.mean(dim=2)  # (B*M), C
        else:
            agg = self.temporal_conv(pose_stack)  # (B*M), C, L
            if agg.shape[-1] > 1:
                agg = agg.mean(dim=2)  # (B*M), C
            else:
                agg = agg.squeeze(-1)  # (B*M), C
        agg = agg.reshape(B, M, C)  # B, M, C

        pred = self.head(agg).reshape(B, M, self.num_joints, 3)  # B, M, J, 3
        pred = pred.mean(dim=1)  # B, J, 3
        return pred
