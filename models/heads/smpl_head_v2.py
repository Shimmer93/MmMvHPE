"""
SMPL Head v2 for predicting SMPL parameters using 6D rotation representation.

This head predicts SMPL parameters:
- global_orient_6d: (B, 6) - root rotation in 6D continuous representation
- body_pose_6d: (B, 23*6) - body joint rotations (23 joints × 6)
- betas: (B, 10) - shape parameters
- transl: (B, 3) - translation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_head import BaseHead


def compute_rotation_matrix_from_6d(poses_6d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrices using Gram-Schmidt.

    Args:
        poses_6d: Tensor of shape (..., 6)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    v1 = poses_6d[..., 0:3]
    v2 = poses_6d[..., 3:6]

    b1 = F.normalize(v1, dim=-1, eps=eps)
    proj = (b1 * v2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(v2 - proj, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def axis_angle_to_matrix(axis_angle: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Convert axis-angle rotations to rotation matrices."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(eps)

    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(axis.shape[:-1] + (3, 3))

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.view((1,) * (axis_angle.dim() - 1) + (3, 3))

    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    rot = eye + sin * K + (1.0 - cos) * torch.matmul(K, K)
    return rot


def rotation_matrix_to_axis_angle(rot_mats: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Convert rotation matrices to axis-angle representation."""
    trace = rot_mats[..., 0, 0] + rot_mats[..., 1, 1] + rot_mats[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos)

    rx = rot_mats[..., 2, 1] - rot_mats[..., 1, 2]
    ry = rot_mats[..., 0, 2] - rot_mats[..., 2, 0]
    rz = rot_mats[..., 1, 0] - rot_mats[..., 0, 1]
    r = torch.stack([rx, ry, rz], dim=-1)

    sin = torch.sin(angle)
    denom = (2.0 * sin).clamp_min(eps).unsqueeze(-1)
    axis = r / denom
    axis_angle = axis * angle.unsqueeze(-1)

    small = sin.abs() < 1e-6
    axis_angle = torch.where(small.unsqueeze(-1), 0.5 * r, axis_angle)
    return axis_angle


def geodesic_distance(rot_pred: torch.Tensor, rot_gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Geodesic distance between rotation matrices using atan2 for stability.
    """
    rot_pred = torch.nan_to_num(rot_pred, nan=0.0, posinf=0.0, neginf=0.0)
    rot_gt = torch.nan_to_num(rot_gt, nan=0.0, posinf=0.0, neginf=0.0)
    rel = torch.matmul(rot_pred, rot_gt.transpose(-1, -2))
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    rx = rel[..., 2, 1] - rel[..., 1, 2]
    ry = rel[..., 0, 2] - rel[..., 2, 0]
    rz = rel[..., 1, 0] - rel[..., 0, 1]
    sin = 0.5 * torch.linalg.norm(torch.stack([rx, ry, rz], dim=-1), dim=-1)
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    sin = torch.clamp(sin, 0.0, 1.0 - eps)
    return torch.atan2(sin, cos)


class SMPLHeadV2(BaseHead):
    """
    SMPL parameter regression head using 6D rotations with geodesic and joint losses.
    """

    NUM_GLOBAL_ORIENT = 6
    NUM_BODY_POSE = 23 * 6
    NUM_TRANSL = 3

    def __init__(
        self,
        losses=None,
        emb_size: int = 512,
        hidden_dims: list = None,
        num_betas: int = 10,
        dropout: float = 0.1,
        activation: str = 'gelu',
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
    ):
        super().__init__(losses or [])

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        self.emb_size = emb_size
        self.num_betas = num_betas
        self.use_smpl_mean = use_smpl_mean
        self.smpl_model_path = smpl_model_path
        self.joint_loss_type = joint_loss_type

        self.rot_weight = rot_weight
        self.root_rot_weight = root_rot_weight
        self.body_rot_weight = body_rot_weight
        self.joint_weight = joint_weight
        self.transl_weight = transl_weight
        self.betas_weight = betas_weight
        self.debug = debug
        self.debug_every = max(1, int(debug_every))
        self._debug_step = 0
        self._debug_warned = False
        self._prev_pred_stats = None
        self.use_simple_rot_loss = use_simple_rot_loss
        self.simple_rot_weight = simple_rot_weight

        self.num_output = self.NUM_GLOBAL_ORIENT + self.NUM_BODY_POSE + num_betas + self.NUM_TRANSL

        layers = []
        in_dim = emb_size

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.GELU

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, self.num_output))
        self.mlp = nn.Sequential(*layers)

        if use_smpl_mean:
            self._init_smpl_mean()

        self._smpl_model = None

    def _is_main_process(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _debug_tensor(self, name, tensor):
        if not self.debug or not self._is_main_process():
            return
        if tensor is None:
            print(f"[SMPLHeadV2][debug] {name}: None")
            return
        is_finite = torch.isfinite(tensor)
        if not is_finite.all():
            num_bad = (~is_finite).sum().item()
            max_abs = tensor.abs().max().item()
            if not self._debug_warned:
                print("[SMPLHeadV2][warn] non-finite values detected; enabling debug output.")
                self._debug_warned = True
            print(f"[SMPLHeadV2][debug] {name}: non-finite={num_bad} max_abs={max_abs:.4e}")

    def _init_smpl_mean(self):
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        mean_global_orient = identity_6d
        mean_body_pose = identity_6d.repeat(23)
        mean_betas = torch.zeros(self.num_betas)
        mean_transl = torch.zeros(self.NUM_TRANSL)

        mean_params = torch.cat([
            mean_global_orient,
            mean_body_pose,
            mean_betas,
            mean_transl,
        ])
        self.register_buffer('mean_params', mean_params)

    @property
    def smpl_model(self):
        if self._smpl_model is None:
            from models.smpl import SMPL
            self._smpl_model = SMPL(model_path=self.smpl_model_path)
        return self._smpl_model

    def _extract_skeleton_token(self, x):
        if isinstance(x, list):
            x = x[-1]
        B, M, T, N, C = x.shape
        x = x[:, :, :, -1, :]
        x = x.mean(dim=[1, 2])
        return x

    def forward(self, x):
        feat = self._extract_skeleton_token(x)
        params = self.mlp(feat)

        if self.use_smpl_mean:
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

    def _parse_gt_smpl(self, gt_smpl, device):
        if isinstance(gt_smpl, list):
            gt_global_orient = torch.stack([
                s['global_orient'] if isinstance(s['global_orient'], torch.Tensor)
                else torch.from_numpy(s['global_orient'])
                for s in gt_smpl
            ]).to(device)
            gt_body_pose = torch.stack([
                s['body_pose'] if isinstance(s['body_pose'], torch.Tensor)
                else torch.from_numpy(s['body_pose'])
                for s in gt_smpl
            ]).to(device)
            gt_betas = torch.stack([
                s['betas'] if isinstance(s['betas'], torch.Tensor)
                else torch.from_numpy(s['betas'])
                for s in gt_smpl
            ]).to(device)
            gt_transl = torch.stack([
                s['transl'] if isinstance(s['transl'], torch.Tensor)
                else torch.from_numpy(s['transl'])
                for s in gt_smpl
            ]).to(device)
        else:
            gt_global_orient = gt_smpl['global_orient'].to(device)
            gt_body_pose = gt_smpl['body_pose'].to(device)
            gt_betas = gt_smpl['betas'].to(device)
            gt_transl = gt_smpl['transl'].to(device)

        return (
            gt_global_orient.float(),
            gt_body_pose.float(),
            gt_betas.float(),
            gt_transl.float(),
        )

    def _pred_pose_axis_angle(self, pred_rot_mats):
        pred_global_rot, pred_body_rot = pred_rot_mats
        pred_global_aa = rotation_matrix_to_axis_angle(pred_global_rot)
        pred_body_aa = rotation_matrix_to_axis_angle(pred_body_rot)
        return pred_global_aa, pred_body_aa

    def loss(self, x, data_batch):
        pred = self.forward(x)
        device = pred['global_orient'].device
        self._debug_step += 1
        do_debug = self.debug and (self._debug_step % self.debug_every == 0)

        gt_smpl = data_batch['gt_smpl']
        gt_global_orient, gt_body_pose, gt_betas, gt_transl = self._parse_gt_smpl(gt_smpl, device)

        B = pred['global_orient'].shape[0]

        pred_global_6d = torch.nan_to_num(pred['global_orient'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        pred_body_6d = torch.nan_to_num(pred['body_pose'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        if do_debug:
            self._debug_tensor("pred_global_6d", pred_global_6d)
            self._debug_tensor("pred_body_6d", pred_body_6d)
            pg_mean = pred_global_6d.mean().item()
            pg_std = pred_global_6d.std(unbiased=False).item()
            pb_mean = pred_body_6d.mean().item()
            pb_std = pred_body_6d.std(unbiased=False).item()
            delta_msg = ""
            if self._prev_pred_stats is not None:
                d_pg_mean = pg_mean - self._prev_pred_stats["pg_mean"]
                d_pg_std = pg_std - self._prev_pred_stats["pg_std"]
                d_pb_mean = pb_mean - self._prev_pred_stats["pb_mean"]
                d_pb_std = pb_std - self._prev_pred_stats["pb_std"]
                delta_msg = (
                    f" delta_global_mean={d_pg_mean:.4e} delta_global_std={d_pg_std:.4e} "
                    f"delta_body_mean={d_pb_mean:.4e} delta_body_std={d_pb_std:.4e}"
                )
            print(
                "[SMPLHeadV2][debug] pred_6d stats "
                f"global_mean={pg_mean:.4e} global_std={pg_std:.4e} "
                f"body_mean={pb_mean:.4e} body_std={pb_std:.4e}{delta_msg}"
            )
            self._prev_pred_stats = {
                "pg_mean": pg_mean,
                "pg_std": pg_std,
                "pb_mean": pb_mean,
                "pb_std": pb_std,
            }
        pred_global_rot = compute_rotation_matrix_from_6d(pred_global_6d.reshape(-1, 6))
        pred_global_rot = pred_global_rot.reshape(B, 3, 3)

        pred_body_rot = compute_rotation_matrix_from_6d(pred_body_6d.reshape(-1, 6))
        pred_body_rot = pred_body_rot.reshape(B, 23, 3, 3)
        if do_debug:
            self._debug_tensor("pred_global_rot", pred_global_rot)
            self._debug_tensor("pred_body_rot", pred_body_rot)

        gt_global_rot = axis_angle_to_matrix(gt_global_orient.float())
        gt_body_rot = axis_angle_to_matrix(gt_body_pose.float().reshape(B, 23, 3))

        rot_root = geodesic_distance(pred_global_rot, gt_global_rot).mean()
        rot_body = geodesic_distance(pred_body_rot, gt_body_rot).mean()
        rot_loss = (self.root_rot_weight * rot_root) + (self.body_rot_weight * rot_body)
        if do_debug:
            self._debug_tensor("rot_root", rot_root)
            self._debug_tensor("rot_body", rot_body)

        simple_rot_loss = torch.zeros((), device=device)
        if self.use_simple_rot_loss:
            gt_global_6d = gt_global_rot[..., :2].reshape(B, 6)
            gt_body_6d = gt_body_rot[..., :2].reshape(B, 23 * 6)
            simple_rot_loss = (
                F.mse_loss(pred_global_6d, gt_global_6d) +
                F.mse_loss(pred_body_6d, gt_body_6d)
            )
            rot_loss = rot_loss + self.simple_rot_weight * simple_rot_loss

        pred_transl = torch.nan_to_num(pred['transl'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        pred_betas = torch.nan_to_num(pred['betas'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        transl_loss = F.l1_loss(pred_transl, gt_transl.float())
        betas_loss = F.mse_loss(pred_betas, gt_betas[:, :self.num_betas].float())
        if do_debug:
            self._debug_tensor("transl_loss", transl_loss)
            self._debug_tensor("betas_loss", betas_loss)

        joint_weight = self.joint_weight
        joint_loss = torch.zeros((), device=device)
        gt_keypoints = data_batch.get('gt_keypoints', None)
        if gt_keypoints is not None:
            if isinstance(gt_keypoints, np.ndarray):
                gt_keypoints = torch.from_numpy(gt_keypoints)
            gt_keypoints = gt_keypoints.to(device).float()
            if gt_keypoints.dim() == 2:
                gt_keypoints = gt_keypoints.unsqueeze(0)

            pred_global_aa, pred_body_aa = self._pred_pose_axis_angle((pred_global_rot, pred_body_rot))
            pred_pose = torch.cat([pred_global_aa, pred_body_aa.reshape(B, -1)], dim=1)
            pred_pose = torch.nan_to_num(pred_pose, nan=0.0, posinf=0.0, neginf=0.0)
            if do_debug:
                self._debug_tensor("pred_pose_axis_angle", pred_pose)

            betas = pred_betas
            expected_betas = self.smpl_model.th_betas.shape[1]
            if betas.shape[1] < expected_betas:
                betas_padded = torch.zeros(B, expected_betas, device=device, dtype=betas.dtype)
                betas_padded[:, :betas.shape[1]] = betas
                betas = betas_padded

            smpl_model = self.smpl_model.to(device)
            transl = pred_transl
            output = smpl_model(pred_pose, betas, transl)
            pred_joints = output[1] if isinstance(output, (list, tuple)) else output
            if do_debug:
                self._debug_tensor("pred_joints", pred_joints)

            if self.joint_loss_type == 'mse':
                joint_loss = F.mse_loss(pred_joints, gt_keypoints)
            else:
                joint_loss = F.l1_loss(pred_joints, gt_keypoints)
            if do_debug:
                self._debug_tensor("joint_loss", joint_loss)
        else:
            joint_weight = 0.0

        if do_debug:
            self._debug_tensor("rot_loss", rot_loss)

        if self.debug and self._is_main_process():
            loss_checks = {
                "rot_loss": rot_loss,
                "joint_loss": joint_loss,
                "transl_loss": transl_loss,
                "betas_loss": betas_loss,
            }
            for name, value in loss_checks.items():
                if not torch.isfinite(value).all():
                    print(f"[SMPLHeadV2][warn] non-finite loss detected: {name}")

        losses = {
            'smpl_rot': (rot_loss, self.rot_weight),
            'smpl_joint': (joint_loss, joint_weight),
            'smpl_transl': (transl_loss, self.transl_weight),
            'smpl_betas': (betas_loss, self.betas_weight),
        }
        return losses

    def predict(self, x):
        pred = self.forward(x)
        B = pred['global_orient'].shape[0]

        pred_global_6d = torch.nan_to_num(pred['global_orient'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        pred_body_6d = torch.nan_to_num(pred['body_pose'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        pred_global_rot = compute_rotation_matrix_from_6d(pred_global_6d.reshape(-1, 6))
        pred_global_rot = pred_global_rot.reshape(B, 3, 3)
        pred_body_rot = compute_rotation_matrix_from_6d(pred_body_6d.reshape(-1, 6))
        pred_body_rot = pred_body_rot.reshape(B, 23, 3, 3)

        pred_global_aa = rotation_matrix_to_axis_angle(pred_global_rot)
        pred_body_aa = rotation_matrix_to_axis_angle(pred_body_rot).reshape(B, -1)
        pred_global_aa = torch.nan_to_num(pred_global_aa, nan=0.0, posinf=0.0, neginf=0.0)
        pred_body_aa = torch.nan_to_num(pred_body_aa, nan=0.0, posinf=0.0, neginf=0.0)

        pred_betas = torch.nan_to_num(pred['betas'], nan=0.0, posinf=0.0, neginf=0.0)
        pred_transl = torch.nan_to_num(pred['transl'], nan=0.0, posinf=0.0, neginf=0.0)
        return {
            'global_orient': pred_global_aa,
            'body_pose': pred_body_aa,
            'betas': pred_betas,
            'transl': pred_transl,
            'global_orient_6d': pred_global_6d,
            'body_pose_6d': pred_body_6d,
        }
