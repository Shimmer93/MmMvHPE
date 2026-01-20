import torch
import torch.nn as nn
from einops import rearrange

from misc.pose_enc import pose_encoding_to_extri_intri
from .base_head import BaseHead


class RegressionKeypointHeadV4(BaseHead):
    def __init__(
        self,
        losses,
        emb_size=512,
        num_joints=24,
        num_register_tokens=4,
        num_smpl_tokens=1,
        max_modalities=4,
        last_n_layers=-1,
        pose_encoding_type="absT_quaR_FoV",
    ):
        super().__init__(losses)
        self.emb_size = emb_size
        self.num_joints = num_joints
        self.num_register_tokens = num_register_tokens
        self.num_smpl_tokens = num_smpl_tokens
        self.max_modalities = max_modalities
        self.last_n_layers = last_n_layers
        self.pose_encoding_type = pose_encoding_type

        self.projector = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, 3),
        )

        global_in = emb_size * max_modalities
        self.global_projector = nn.Sequential(
            nn.LayerNorm(global_in),
            nn.Linear(global_in, emb_size),
            nn.ReLU(),
        )
        self.global_norm = nn.LayerNorm(emb_size)
        self.global_mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size // 2),
            nn.ReLU(),
            nn.Linear(emb_size // 2, 3),
        )

    def forward(self, x):
        x = self._select_layers(x)
        x = x[..., x.shape[-1] // 2 :]
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B, T, M, S, C = x.shape
        joint_tokens = x[..., S - self.num_joints :, :]
        joint_tokens = joint_tokens.mean(dim=1)

        pred_per_modality = self._regress_per_modality(joint_tokens)
        pred_global = self._regress_global(joint_tokens)

        return {"per_modality": pred_per_modality, "global": pred_global}

    def loss(self, x, data_batch):
        outputs = self.forward(x)
        pred_per_modality = outputs["per_modality"]
        pred_global = outputs["global"]

        gt_keypoints = data_batch.get("gt_keypoints", None)
        if gt_keypoints is None:
            return {}

        if isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = gt_keypoints.float()
        else:
            gt_keypoints = torch.as_tensor(gt_keypoints, dtype=torch.float32)
        if gt_keypoints.dim() == 2:
            gt_keypoints = gt_keypoints.unsqueeze(0)
        gt_keypoints = gt_keypoints.to(pred_global.device)

        modalities = data_batch.get("modalities", [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]

        losses = {}
        num_modalities = min(pred_per_modality.shape[1], len(modalities))
        for i in range(num_modalities):
            modality = modalities[i]
            proj_pred, proj_gt = self._project_keypoints(
                pred_per_modality[:, i], gt_keypoints, modality, data_batch
            )
            if proj_pred is None:
                continue
            for loss_name, (loss_fn, loss_weight) in self.losses.items():
                losses[f"{loss_name}_{modality}"] = (loss_fn(proj_pred, proj_gt), loss_weight)

        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[f"{loss_name}_global"] = (loss_fn(pred_global, gt_keypoints), loss_weight)

        return losses

    def predict(self, x):
        outputs = self.forward(x)
        return outputs["global"]

    def _select_layers(self, x):
        if not isinstance(x, list):
            return x
        if self.last_n_layers > 0:
            x = x[-self.last_n_layers :]
        return torch.cat(x, dim=-1)

    def _regress_per_modality(self, joint_tokens):
        B, M, J, C = joint_tokens.shape
        feats = joint_tokens.reshape(B * M, J, C)
        feats = self.projector(feats)
        feats = self.norm(feats)
        pred = self.mlp(feats)
        return pred.reshape(B, M, J, 3)

    def _regress_global(self, joint_tokens):
        B, M, J, C = joint_tokens.shape
        if M < self.max_modalities:
            pad = torch.zeros(B, self.max_modalities - M, J, C, device=joint_tokens.device, dtype=joint_tokens.dtype)
            joint_tokens = torch.cat([joint_tokens, pad], dim=1)
        elif M > self.max_modalities:
            joint_tokens = joint_tokens[:, : self.max_modalities, :, :]

        feats = rearrange(joint_tokens, "b m j c -> b j (m c)")
        feats = self.global_projector(feats)
        feats = self.global_norm(feats)
        return self.global_mlp(feats)

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
        cam_points = RegressionKeypointHeadV4._transform_to_camera(points, extrinsics)
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
