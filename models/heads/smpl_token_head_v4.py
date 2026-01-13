"""
SMPL token head with iterative refinement (VIBE-style) over aggregator tokens.

Token layout expected from aggregator:
  [camera, smpl_global, smpl_betas, smpl_transl, joints...]
Joints include the root joint at index 0; body_pose uses joints[1:].
"""

import torch
import torch.nn as nn

from .smpl_head_v2 import SMPLHeadV2


class SMPLTokenHeadV4(SMPLHeadV2):
    """
    Iteratively refine SMPL parameters from SMPL/joint tokens using 6D rotations.
    """
    def __init__(
        self,
        losses=None,
        emb_size: int = 512,
        num_betas: int = 10,
        dropout: float = 0.1,
        use_smpl_mean: bool = True,
        smpl_model_path: str = "weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        joint_loss_type: str = "l1",
        rot_weight: float = 1.0,
        root_rot_weight: float = 10.0,
        body_rot_weight: float = 1.0,
        joint_weight: float = 1.0,
        transl_weight: float = 1.0,
        betas_weight: float = 1.0,
        debug: bool = False,
        debug_every: int = 100,
        use_simple_rot_loss: bool = False,
        simple_rot_weight: float = 1.0,
        num_joints: int = 24,
        num_smpl_tokens: int = 3,
        n_iter: int = 3,
        hidden_dim: int = 1024,
    ):
        super().__init__(
            losses=losses,
            emb_size=emb_size,
            hidden_dims=None,
            num_betas=num_betas,
            dropout=dropout,
            activation="gelu",
            use_smpl_mean=use_smpl_mean,
            smpl_model_path=smpl_model_path,
            joint_loss_type=joint_loss_type,
            rot_weight=rot_weight,
            root_rot_weight=root_rot_weight,
            body_rot_weight=body_rot_weight,
            joint_weight=joint_weight,
            transl_weight=transl_weight,
            betas_weight=betas_weight,
            debug=debug,
            debug_every=debug_every,
            use_simple_rot_loss=use_simple_rot_loss,
            simple_rot_weight=simple_rot_weight,
        )

        self.num_joints = num_joints
        self.num_smpl_tokens = num_smpl_tokens
        self.n_iter = max(1, int(n_iter))

        self.input_norm = nn.LayerNorm(emb_size)
        self.smpl_pool = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 1),
        )
        self.joint_pool = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 1),
        )

        context_dim = emb_size * 3
        self.fc1 = nn.Linear(context_dim + self.num_output, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.dec = nn.Linear(hidden_dim, self.num_output)

        nn.init.xavier_uniform_(self.dec.weight, gain=0.01)
        if self.dec.bias is not None:
            nn.init.constant_(self.dec.bias, 0)

    @staticmethod
    def _attn_pool(tokens, scorer):
        weights = scorer(tokens).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)

    def _init_params(self, batch_size, device, dtype):
        if self.use_smpl_mean:
            init = self.mean_params.to(device=device, dtype=dtype)
            init = init.unsqueeze(0).expand(batch_size, -1)
        else:
            init = torch.zeros(batch_size, self.num_output, device=device, dtype=dtype)
        return init

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]

        B, M, T, N, C = x.shape
        x = x.mean(dim=[1, 2])
        x = self.input_norm(x)

        if self.num_smpl_tokens < 3:
            raise ValueError("num_smpl_tokens must be at least 3 for global/betas/transl.")

        smpl_start = 1
        smpl_end = smpl_start + self.num_smpl_tokens
        joint_start = smpl_end
        joint_end = joint_start + self.num_joints

        if N < joint_end:
            raise ValueError(
                f"Expected at least {joint_end} tokens, got {N}. "
                "Check aggregator token layout."
            )

        camera_token = x[:, 0:1, :]
        smpl_tokens = x[:, smpl_start:smpl_end, :]
        joint_tokens = x[:, joint_start:joint_end, :]

        smpl_context = self._attn_pool(smpl_tokens, self.smpl_pool)
        joint_context = self._attn_pool(joint_tokens, self.joint_pool)
        context = torch.cat([camera_token.squeeze(1), smpl_context, joint_context], dim=1)

        params = self._init_params(B, x.device, x.dtype)
        for _ in range(self.n_iter):
            xc = torch.cat([context, params], dim=1)
            xc = self.drop1(torch.relu(self.fc1(xc)))
            xc = self.drop2(torch.relu(self.fc2(xc)))
            params = params + self.dec(xc)

        idx = 0
        global_orient = params[:, idx:idx + self.NUM_GLOBAL_ORIENT]
        idx += self.NUM_GLOBAL_ORIENT
        body_pose = params[:, idx:idx + self.NUM_BODY_POSE]
        idx += self.NUM_BODY_POSE
        betas = params[:, idx:idx + self.num_betas]
        idx += self.num_betas
        transl = params[:, idx:idx + self.NUM_TRANSL]

        return {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "betas": betas,
            "transl": transl,
        }
