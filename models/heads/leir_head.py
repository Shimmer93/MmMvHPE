import torch
import torch.nn as nn

from models.leir.smpl import LEIRSMPL

from .base_head import BaseHead


class LEIRHead(BaseHead):
    def __init__(self, losses, smpl_model_path: str = 'weights/smpl/SMPL_NEUTRAL.pkl'):
        super().__init__(losses)
        self.smpl = LEIRSMPL(smpl_model_path)
        self._smpl_device = None

    def forward(self, x):
        return x

    def loss(self, x, data_batch):
        pred_output = self.forward(x)
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            loss_dict = loss_fn(pred_output, data_batch)
            for key, value in loss_dict.items():
                if key == 'total_loss':
                    continue
                losses[key] = (value, loss_weight)
        return losses

    def predict(self, x):
        pred = self.forward(x)
        pred_keypoints = pred.get('img_kp_3d', pred.get('joint'))
        if pred_keypoints is not None and pred_keypoints.dim() == 4:
            pred_keypoints = pred_keypoints[:, -1]
        pred_smpl_params = pred.get('img_theta')
        if pred_smpl_params is not None and pred_smpl_params.dim() == 3:
            pred_smpl_params = pred_smpl_params[:, -1]
        pred_camera = None
        if pred_smpl_params is not None and pred_smpl_params.shape[-1] >= 85:
            pred_camera = pred_smpl_params[..., :3]
            pred_smpl_params = pred_smpl_params[..., 3:85]
        if pred_smpl_params is not None and pred_smpl_params.shape[-1] >= 72:
            pred_smpl_params = pred_smpl_params.clone()
            pred_smpl_params[..., :3] = 0.0
        pred_keypoints_raw = pred_keypoints
        pred_smpl_keypoints = None
        if pred_smpl_params is not None and pred_smpl_params.shape[-1] >= 82:
            pose = pred_smpl_params[..., :72]
            betas = pred_smpl_params[..., 72:82]
            if self._smpl_device != pose.device:
                self.smpl = self.smpl.to(pose.device)
                self._smpl_device = pose.device
            vertices = self.smpl(pose, betas)
            pred_smpl_keypoints = self.smpl.get_full_joints(vertices)
        pred_smpl = None
        if pred_smpl_params is not None and pred_smpl_params.shape[-1] >= 82:
            pose = pred_smpl_params[..., :72]
            betas = pred_smpl_params[..., 72:82]
            batch_shape = pose.shape[:-1]
            device = pose.device
            global_orient = torch.zeros((*batch_shape, 3), device=device, dtype=pose.dtype)
            body_pose = pose[..., 3:72]
            transl = torch.zeros((*batch_shape, 3), device=device, dtype=pose.dtype)
            pred_smpl = {
                'global_orient': global_orient,
                'body_pose': body_pose,
                'betas': betas,
                'transl': transl,
            }
        if pred_smpl_keypoints is not None:
            pred_keypoints = pred_smpl_keypoints
        return {
            'pred_smpl_params': pred_smpl_params,
            'pred_keypoints': pred_keypoints,
            'pred_smpl_keypoints': pred_smpl_keypoints,
            'pred_camera': pred_camera,
            'pred_smpl': pred_smpl,
            'pred_keypoints_raw': pred_keypoints_raw,
            **pred,
        }
