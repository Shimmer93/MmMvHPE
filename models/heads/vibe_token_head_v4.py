import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from models.smpl import SMPL
from misc.pose_enc import pose_encoding_to_extri_intri
from misc.rotation import rot6d_to_rotmat, rotation_matrix_to_angle_axis, batch_rodrigues
from .base_head import BaseHead


class Regressor(nn.Module):
    def __init__(self, smpl_path, smpl_mean_params, emb_size=1024):
        super(Regressor, self).__init__()

        npose = 24 * 6
        self.smpl = SMPL(model_path=smpl_path)

        self.joint_fc1 = nn.Linear(emb_size + 6, emb_size)
        self.joint_fc2 = nn.Linear(emb_size, emb_size)
        self.decpose = nn.Linear(emb_size, 6)
        self.shape_fc1 = nn.Linear(emb_size + 10, emb_size)
        self.shape_fc2 = nn.Linear(emb_size, emb_size)
        self.decshape = nn.Linear(emb_size, 10)
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

    def forward(self, x, init_pose=None, init_shape=None, n_iter=3):
        device = x.device
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B, N, C), got {tuple(x.shape)}.")
        batch_size, num_tokens, _ = x.shape
        num_smpl_tokens = 1
        num_joints = 24
        if num_tokens != num_smpl_tokens + num_joints:
            raise ValueError(
                f"Expected {num_smpl_tokens + num_joints} tokens, got {num_tokens}."
            )

        smpl_tokens = x[:, :num_smpl_tokens, :]
        joint_tokens = x[:, num_smpl_tokens:, :]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)

        pred_pose = init_pose.view(batch_size, num_joints, 6)
        pred_shape = init_shape

        for _ in range(n_iter):
            joint_x = torch.cat([joint_tokens, pred_pose], dim=-1)
            joint_x = self.joint_fc1(joint_x)
            joint_x = self.drop1(joint_x)
            joint_x = self.joint_fc2(joint_x)
            joint_x = self.drop2(joint_x)

            smpl_x = torch.cat([smpl_tokens, pred_shape.unsqueeze(1)], dim=-1)
            smpl_x = self.shape_fc1(smpl_x)
            smpl_x = self.drop1(smpl_x)
            smpl_x = self.shape_fc2(smpl_x)
            smpl_x = self.drop2(smpl_x)

            pred_pose = self.decpose(joint_x) + pred_pose
            pred_shape = self.decshape(smpl_x.squeeze(1)) + pred_shape

        pred_pose_flat = pred_pose.reshape(batch_size, -1)
        pred_rotmat = rot6d_to_rotmat(pred_pose_flat).view(batch_size, 24, 3, 3)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        pose[:, :3] = 0.0

        smpl_model = self.smpl.to(device)
        output = smpl_model(
            pose,
            pred_shape,
            torch.zeros((batch_size, 3)).to(device),
        )
        pred_joints = output[1] if isinstance(output, (list, tuple)) else output

        output = {
            'pred_smpl_params': torch.cat([pose, pred_shape], dim=1),
            'pred_keypoints': pred_joints,
            'pred_rotmat': pred_rotmat,
        }
        return output


class VIBETokenHeadV4(BaseHead):
    def __init__(
        self,
        losses,
        smpl_path,
        smpl_mean_params,
        emb_size,
        num_register_tokens=4,
        num_smpl_tokens=1,
        max_modalities=4,
        n_iters=3,
        last_n_layers=-1,
        pose_encoding_type="absT_quaR_FoV",
    ):
        super().__init__(losses)
        self.emb_size = emb_size
        self.num_register_tokens = num_register_tokens
        self.num_smpl_tokens = num_smpl_tokens
        self.max_modalities = max_modalities
        self.n_iters = n_iters
        self.last_n_layers = last_n_layers
        self.pose_encoding_type = pose_encoding_type

        self.projector = nn.Sequential(
            nn.LayerNorm(emb_size * max_modalities),
            nn.Linear(emb_size * max_modalities, emb_size),
            nn.ReLU(),
        )

        self.regressor = Regressor(smpl_path, smpl_mean_params, emb_size=emb_size)

    def forward(self, x):
        x = self._select_layers(x)
        x = x[..., : x.shape[-1] // 2]
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B, T, M, S, C = x.shape
        num_special = 1 + self.num_register_tokens
        start = num_special
        end = num_special + self.num_smpl_tokens + 24
        tokens = x[:, -1, :, start:end, :]  # (B, M, num_smpl_tokens + 24, C)

        if M < self.max_modalities:
            pad = torch.zeros(
                B, self.max_modalities - M, tokens.shape[2], C,
                device=tokens.device, dtype=tokens.dtype
            )
            tokens = torch.cat([tokens, pad], dim=1)
        elif M > self.max_modalities:
            tokens = tokens[:, : self.max_modalities, :, :]

        tokens = rearrange(tokens, 'b m n c -> b n (m c)')
        tokens = self.projector(tokens)
        output = self.regressor(tokens, n_iter=self.n_iters)

        return {
            'pred_smpl_params': output['pred_smpl_params'],
            'pred_keypoints': output['pred_keypoints'],
            'pred_rotmat': output['pred_rotmat'],
        }

    def loss(self, x, data_batch):
        pred_output = self.forward(x)
        losses = {}
        gt_rotmat = data_batch.get('gt_rotmat', None)
        if gt_rotmat is None:
            gt_pose = data_batch['gt_smpl_params'][:, :72].contiguous()
            gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
            data_batch['gt_rotmat'] = gt_rotmat

        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            if 'keypoint' in loss_name.lower():
                losses[loss_name] = (loss_fn(pred_output['pred_keypoints'], data_batch['gt_keypoints']), loss_weight)
            elif 'smplpose' in loss_name.lower():
                losses[loss_name] = (
                    loss_fn(pred_output['pred_smpl_params'][:, :72], data_batch['gt_smpl_params'][:, :72]),
                    loss_weight,
                )
            elif 'smplshape' in loss_name.lower():
                losses[loss_name] = (
                    loss_fn(pred_output['pred_smpl_params'][:, 72:], data_batch['gt_smpl_params'][:, 72:]),
                    loss_weight,
                )
            elif 'rotmat' in loss_name.lower():
                losses[loss_name] = (loss_fn(pred_output['pred_rotmat'], data_batch['gt_rotmat']), loss_weight)

        gt_keypoints = data_batch.get("gt_keypoints", None)
        if gt_keypoints is not None:
            if isinstance(gt_keypoints, torch.Tensor):
                gt_keypoints = gt_keypoints.float()
            else:
                gt_keypoints = torch.as_tensor(gt_keypoints, dtype=torch.float32)
            if gt_keypoints.dim() == 2:
                gt_keypoints = gt_keypoints.unsqueeze(0)
            gt_keypoints = gt_keypoints.to(pred_output['pred_keypoints'].device)

            modalities = data_batch.get("modalities", [])
            if modalities and isinstance(modalities[0], (list, tuple)):
                modalities = modalities[0]

            for modality in modalities:
                proj_pred, proj_gt = self._project_keypoints(
                    pred_output['pred_keypoints'], gt_keypoints, modality, data_batch
                )
                if proj_pred is None:
                    continue
                for loss_name, (loss_fn, loss_weight) in self.losses.items():
                    if 'keypoint' not in loss_name.lower():
                        continue
                    losses[f"{loss_name}_{modality}_proj"] = (loss_fn(proj_pred, proj_gt), loss_weight)
                for loss_name, (loss_fn, loss_weight) in self.losses.items():
                    if 'smplpose' in loss_name.lower():
                        losses[f"{loss_name}_{modality}"] = (
                            loss_fn(pred_output['pred_smpl_params'][:, :72], data_batch['gt_smpl_params'][:, :72]),
                            loss_weight,
                        )
                    elif 'smplshape' in loss_name.lower():
                        losses[f"{loss_name}_{modality}"] = (
                            loss_fn(pred_output['pred_smpl_params'][:, 72:], data_batch['gt_smpl_params'][:, 72:]),
                            loss_weight,
                        )
                    elif 'rotmat' in loss_name.lower():
                        losses[f"{loss_name}_{modality}"] = (
                            loss_fn(pred_output['pred_rotmat'], data_batch['gt_rotmat']),
                            loss_weight,
                        )

        return losses

    def predict(self, x):
        return self.forward(x)

    def _select_layers(self, x):
        if not isinstance(x, list):
            return x
        if self.last_n_layers > 0:
            x = x[-self.last_n_layers :]
        return torch.cat(x, dim=-1)

    def _project_keypoints(self, pred, gt, modality, data_batch):
        modality = modality.lower()
        cam_params = self._get_camera_params(data_batch, modality, pred.device)
        if cam_params is None:
            return None, None
        extrinsics, intrinsics, image_size = cam_params

        if modality in {"rgb", "depth"}:
            pred_proj = self._project_to_image(pred, extrinsics, intrinsics)
            gt_proj = self._project_to_image(gt, extrinsics, intrinsics)
            pred_proj = self._normalize_2d(pred_proj, image_size)
            gt_proj = self._normalize_2d(gt_proj, image_size)
            pred_proj = torch.clamp(pred_proj, -1.0, 1.0)
            gt_proj = torch.clamp(gt_proj, -1.0, 1.0)
            return pred_proj, gt_proj
        if modality in {"lidar", "mmwave"}:
            pred_cam = self._transform_to_camera(pred, extrinsics)
            gt_cam = self._transform_to_camera(gt, extrinsics)
            return pred_cam, gt_cam

        return None, None

    def _get_camera_params(self, data_batch, modality, device):
        gt_camera = data_batch.get(f"gt_camera_{modality}", None)
        if gt_camera is None:
            return None
        if not isinstance(gt_camera, torch.Tensor):
            gt_camera = torch.as_tensor(gt_camera, dtype=torch.float32)
        gt_camera = gt_camera.to(device).float()
        if gt_camera.dim() == 2:
            gt_camera = gt_camera.unsqueeze(0)
        if gt_camera.dim() == 3:
            gt_camera = gt_camera[:, -1]

        image_size = self._get_image_size(data_batch, modality)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            gt_camera.unsqueeze(1),
            image_size_hw=image_size,
            pose_encoding_type=self.pose_encoding_type,
            build_intrinsics=True,
        )
        return extrinsics.squeeze(1), intrinsics.squeeze(1), image_size

    def _get_image_size(self, data_batch, modality):
        input_key = f"input_{modality}"
        if input_key in data_batch:
            inp = data_batch[input_key]
            if isinstance(inp, torch.Tensor) and inp.dim() >= 4:
                return int(inp.shape[-2]), int(inp.shape[-1])
        return (224, 224)

    @staticmethod
    def _transform_to_camera(points, extrinsics):
        R = extrinsics[:, :3, :3]
        T = extrinsics[:, :3, 3]
        return torch.einsum("bij,bkj->bki", R, points) + T.unsqueeze(1)

    @staticmethod
    def _project_to_image(points, extrinsics, intrinsics):
        cam_points = VIBETokenHeadV4._transform_to_camera(points, extrinsics)
        cam_z = cam_points[..., 2].clamp(min=1e-6)
        proj = torch.einsum("bij,bkj->bki", intrinsics, cam_points)
        u = proj[..., 0] / cam_z
        v = proj[..., 1] / cam_z
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _normalize_2d(points_2d, image_size_hw):
        height, width = image_size_hw
        x = points_2d[..., 0] / (width - 1) * 2.0 - 1.0
        y = points_2d[..., 1] / (height - 1) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)
