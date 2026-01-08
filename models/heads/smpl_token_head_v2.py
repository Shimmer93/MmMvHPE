"""
SMPL token head for predicting SMPL parameters using dedicated tokens.

Token layout expected from aggregator:
  [camera, smpl_global, smpl_betas, smpl_transl, joints...]
Joints include the root joint at index 0; body_pose uses joints[1:].
"""

import torch
import torch.nn as nn

from .smpl_head_v2 import SMPLHeadV2


class SMPLTokenHeadV2(SMPLHeadV2):
    """
    Predict SMPL parameters from SMPL/joint tokens using 6D rotations.
    """
    def __init__(
        self,
        losses=None,
        emb_size: int = 512,
        num_betas: int = 10,
        dropout: float = 0.1,
        use_smpl_mean: bool = True,
        smpl_model_path: str = 'weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
        joint_loss_type: str = 'l1',
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
    ):
        super().__init__(
            losses=losses,
            emb_size=emb_size,
            hidden_dims=None,
            num_betas=num_betas,
            dropout=dropout,
            activation='gelu',
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

        self.norm = nn.LayerNorm(emb_size)
        self.global_head = nn.Linear(emb_size, self.NUM_GLOBAL_ORIENT)
        self.body_pose_head = nn.Linear(emb_size, 6)
        self.betas_head = nn.Linear(emb_size, num_betas)
        self.transl_head = nn.Linear(emb_size, self.NUM_TRANSL)

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]

        B, M, T, N, C = x.shape
        x = x.mean(dim=[1, 2])
        x = self.norm(x)

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

        smpl_tokens = x[:, smpl_start:smpl_end, :]
        joint_tokens = x[:, joint_start:joint_end, :]

        global_orient = self.global_head(smpl_tokens[:, 0, :])
        betas = self.betas_head(smpl_tokens[:, 1, :])
        transl = self.transl_head(smpl_tokens[:, 2, :])

        body_tokens = joint_tokens[:, 1:, :]
        body_pose = self.body_pose_head(body_tokens).reshape(B, -1)

        if self.use_smpl_mean:
            if body_pose.shape[1] != self.NUM_BODY_POSE:
                raise ValueError(
                    f"body_pose size {body_pose.shape[1]} does not match "
                    f"expected {self.NUM_BODY_POSE}."
                )
            params = torch.cat([global_orient, body_pose, betas, transl], dim=1)
            params = params + self.mean_params

            idx = 0
            global_orient = params[:, idx:idx + self.NUM_GLOBAL_ORIENT]
            idx += self.NUM_GLOBAL_ORIENT
            body_pose = params[:, idx:idx + self.NUM_BODY_POSE]
            idx += self.NUM_BODY_POSE
            betas = params[:, idx:idx + self.num_betas]
            idx += self.num_betas
            transl = params[:, idx:idx + self.NUM_TRANSL]

        return {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'betas': betas,
            'transl': transl,
        }
