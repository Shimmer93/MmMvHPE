import torch
import torch.nn as nn


class FusionFormerModalityAggregator(nn.Module):
    def __init__(
        self,
        num_joints: int = 24,
        joint_embed_dim: int = 128,
        pose_embed_dim: int = 256,
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
        self.pose_proj = nn.Linear(joint_embed_dim, pose_embed_dim)

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
        if pose.dim() != 4:
            raise ValueError(f"Expected pose tensor with shape (B, T, J, D), got {pose.shape}")
        B, T, J, D = pose.shape
        if J != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {J}")
        embed = self.rgb_embed if is_rgb else self.depth_embed
        pose = embed(pose)
        pose = self.pose_proj(pose)
        pose = pose.mean(dim=2)
        return pose

    def forward(self, features, **kwargs):
        features_rgb, features_depth, features_lidar, features_mmwave = features
        pose_rgb = self._embed_pose(features_rgb, is_rgb=True) if features_rgb is not None else None
        pose_depth = self._embed_pose(features_depth, is_rgb=False) if features_depth is not None else None

        poses = []
        if pose_rgb is not None:
            poses.append(pose_rgb)
        if pose_depth is not None:
            poses.append(pose_depth)

        if not poses:
            raise ValueError("FusionFormerModalityAggregator requires at least one pose modality.")

        pose_stack = torch.stack(poses, dim=1)
        B, M, T, C = pose_stack.shape
        tokens = pose_stack.reshape(B, M * T, C)

        if M * T > self.pos_enc.shape[1]:
            raise ValueError(f"pos_enc length {self.pos_enc.shape[1]} < tokens {M*T}")

        tokens = self.pre_ln(tokens + self.pos_enc[:, : M * T])

        for _ in range(self.num_blocks):
            global_feat = self.encoder(tokens)
            fused = []
            for m in range(M):
                tgt = pose_stack[:, m]
                tgt = self.dec_ln(tgt + self.pos_dec[:, :T])
                dec_out = self.decoder(tgt, global_feat)
                fused.append(dec_out)
            pose_stack = torch.stack(fused, dim=1)
            tokens = pose_stack.reshape(B, M * T, C)

        pose_stack = pose_stack.reshape(B * M, T, C).transpose(1, 2)
        agg = self.temporal_conv(pose_stack).squeeze(-1)
        agg = agg.reshape(B, M, C)

        pred = self.head(agg).reshape(B, M, self.num_joints, 3)
        pred = pred.mean(dim=1)
        return pred
