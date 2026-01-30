import torch
import torch.nn as nn

from misc.rotation import batch_rodrigues
from models.leir.smpl import LEIRSMPL


class LEIRLoss(nn.Module):
    def __init__(self, smpl_model_path: str = 'weights/smpl/SMPL_NEUTRAL.pkl'):
        super().__init__()
        self.criterion_pose = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_keypoints = nn.MSELoss()
        self.criterion_regr = nn.MSELoss()
        self.smpl = LEIRSMPL(smpl_model_path)
        self._nan_reported = False
        print(f"[DEBUG]: LEIRLoss initialized with SMPL model path: {smpl_model_path}")

    def _log_tensor_stats(self, name, tensor):
        if tensor is None:
            print(f"[LEIRLoss] {name}: None")
            return
        if not torch.is_tensor(tensor):
            print(f"[LEIRLoss] {name}: not a tensor")
            return
        flat = tensor.detach().reshape(-1).float()
        finite = torch.isfinite(flat)
        if finite.any():
            stats = (
                flat[finite].min().item(),
                flat[finite].max().item(),
                flat[finite].mean().item(),
            )
            finite_ratio = finite.float().mean().item()
            print(f"[LEIRLoss] {name}: min={stats[0]:.4e} max={stats[1]:.4e} mean={stats[2]:.4e} finite={finite_ratio:.4f}")
        else:
            print(f"[LEIRLoss] {name}: no finite values")

    def forward(self, preds, gt):
        device = preds['rotmat'].device
        if next(self.smpl.buffers()).device != device:
            self.smpl = self.smpl.to(device)
        if not self._nan_reported:
            def _has_nonfinite(t):
                if t is None or not torch.is_tensor(t):
                    return False
                return not torch.isfinite(t).all().item()
            nonfinite_inputs = any([
                _has_nonfinite(preds.get('img_theta')),
                _has_nonfinite(preds.get('img_kp_3d')),
                _has_nonfinite(preds.get('img_kp_2d')),
                _has_nonfinite(preds.get('rotmat')),
                _has_nonfinite(preds.get('joint')),
                _has_nonfinite(gt.get('joint_3d_cam')),
                _has_nonfinite(gt.get('joint_3d_pc')),
                _has_nonfinite(gt.get('joint_2d')),
                _has_nonfinite(gt.get('pose')),
                _has_nonfinite(gt.get('shape')),
            ])
            if nonfinite_inputs:
                should_log = True
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    should_log = torch.distributed.get_rank() == 0
                if should_log:
                    print("[LEIRLoss] Non-finite values detected in inputs, dumping tensor stats...")
                    self._log_tensor_stats("preds/img_theta", preds.get('img_theta'))
                    self._log_tensor_stats("preds/img_kp_3d", preds.get('img_kp_3d'))
                    self._log_tensor_stats("preds/img_kp_2d", preds.get('img_kp_2d'))
                    self._log_tensor_stats("preds/rotmat", preds.get('rotmat'))
                    self._log_tensor_stats("preds/joint", preds.get('joint'))
                    self._log_tensor_stats("gt/joint_3d_cam", gt.get('joint_3d_cam'))
                    self._log_tensor_stats("gt/joint_3d_pc", gt.get('joint_3d_pc'))
                    self._log_tensor_stats("gt/joint_2d", gt.get('joint_2d'))
                    self._log_tensor_stats("gt/pose", gt.get('pose'))
                    self._log_tensor_stats("gt/shape", gt.get('shape'))
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({"leir_nonfinite_inputs": 1.0}, commit=False)
                    except Exception:
                        pass
                self._nan_reported = True
        gt_joint_3d_cam = gt['joint_3d_cam']
        B, T = gt_joint_3d_cam.shape[:2]

        loss_joints = self.criterion_joints(preds['joint'], gt_joint_3d_cam)

        gt_pose = gt['pose']
        gt_rotmats = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(B, T, 24, 3, 3)
        loss_pose = self.criterion_pose(preds['rotmat'], gt_rotmats)

        pred_vertices = self.smpl(preds['rotmat'].reshape(-1, 24, 3, 3),
                                  torch.zeros((B * T, 10), device=preds['rotmat'].device))
        pred_smpl_joints = self.smpl.get_full_joints(pred_vertices).reshape(B, T, 24, 3)
        loss_smpl_joints = self.criterion_joints(pred_smpl_joints, gt['joint_3d_pc'])

        loss_kp_3d = self.keypoint_3d_loss(preds['img_kp_3d'], gt_joint_3d_cam)
        loss_kp_2d = self.criterion_keypoints(preds['img_kp_2d'], gt['joint_2d'])

        pred_rotmats = batch_rodrigues(preds['img_theta'][:, :, 3:75].reshape(-1, 3)).reshape(B, T, 24, 3, 3)
        loss_pose_2d = self.criterion_regr(pred_rotmats, gt_rotmats)
        loss_shape_2d = self.criterion_regr(preds['img_theta'][:, :, 75:], gt['shape'])

        loss_dict = {
            'loss_joint': loss_joints, # This becomes nan in https://wandb.ai/mmhpe/leir/runs/nz5hjrj3?nw=nwuseryzhanghe
            'loss_pose': loss_pose,
            'loss_smpl_joint': loss_smpl_joints,
            'loss_kp_3d': loss_kp_3d,
            'loss_kp_2d': loss_kp_2d,
            'loss_pose_2d': loss_pose_2d,
            'loss_shape_2d': loss_shape_2d,
        }
        loss_dict['total_loss'] = torch.stack(list(loss_dict.values())).sum()

        if not self._nan_reported:
            has_nan = any(torch.isnan(v).any() for v in loss_dict.values())
            if has_nan:
                should_log = True
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    should_log = torch.distributed.get_rank() == 0
                if should_log:
                    print("[LEIRLoss] NaN detected in loss_dict, dumping tensor stats...")
                    self._log_tensor_stats("preds/img_theta", preds.get('img_theta'))
                    self._log_tensor_stats("preds/img_kp_3d", preds.get('img_kp_3d'))
                    self._log_tensor_stats("preds/img_kp_2d", preds.get('img_kp_2d'))
                    self._log_tensor_stats("preds/rotmat", preds.get('rotmat'))
                    self._log_tensor_stats("preds/joint", preds.get('joint'))
                    self._log_tensor_stats("gt/joint_3d_cam", gt.get('joint_3d_cam'))
                    self._log_tensor_stats("gt/joint_3d_pc", gt.get('joint_3d_pc'))
                    self._log_tensor_stats("gt/joint_2d", gt.get('joint_2d'))
                    self._log_tensor_stats("gt/pose", gt.get('pose'))
                    self._log_tensor_stats("gt/shape", gt.get('shape'))
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                "leir_nan_detected": 1.0,
                                "leir_loss_joint": loss_joints.detach().float().item(),
                                "leir_loss_pose": loss_pose.detach().float().item(),
                                "leir_loss_smpl_joint": loss_smpl_joints.detach().float().item(),
                                "leir_loss_kp_3d": loss_kp_3d.detach().float().item(),
                                "leir_loss_kp_2d": loss_kp_2d.detach().float().item(),
                                "leir_loss_pose_2d": loss_pose_2d.detach().float().item(),
                                "leir_loss_shape_2d": loss_shape_2d.detach().float().item(),
                            }, commit=False)
                    except Exception:
                        pass
                self._nan_reported = True
        return loss_dict

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, :, 0:1, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, :, 0:1, :]
        return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)
