import numpy as np
import torch
import torch.nn.functional as F

from misc.pose_enc import pose_encoding_to_extri_intri
from .base_head import BaseHead
from .vggt_camera_head import CameraHead


class VGGTCameraHeadV5(BaseHead):
    def __init__(
        self,
        losses,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        last_n_layers: int = -1,
        feature_slice: str = "full",
        proj_loss_weight_rgb: float = 1.0,
        proj_loss_weight_lidar: float = 1.0,
        proj_loss_type: str = "l1",
    ):
        super().__init__(losses)
        self.camera_head = CameraHead(
            dim_in=dim_in,
            trunk_depth=trunk_depth,
            pose_encoding_type=pose_encoding_type,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            trans_act=trans_act,
            quat_act=quat_act,
            fl_act=fl_act,
        )
        self.pose_encoding_type = pose_encoding_type
        self.last_n_layers = last_n_layers
        self.feature_slice = feature_slice
        self.proj_loss_weight_rgb = proj_loss_weight_rgb
        self.proj_loss_weight_lidar = proj_loss_weight_lidar
        self.proj_loss_type = proj_loss_type

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        x = self._select_layers(aggregated_tokens_list)

        if x.dim() == 5:
            x = x[:, -1, :, :1, :]  # B, M, 1, C (camera token)
        elif x.dim() == 4:
            num_camera_tokens = x.shape[2] - 1 - 24
            x = x[:, -1, :num_camera_tokens, :]
            x = x.unsqueeze(-2)
        else:
            raise ValueError(f"Unexpected token shape: {tuple(x.shape)}")

        return self.camera_head([x], num_iterations)

    def loss(self, x, data_batch):
        pred_camera_enc_list = self.forward(x, num_iterations=data_batch.get("num_camera_iterations", 4))
        pred_camera_encs = pred_camera_enc_list[-1]

        modalities = data_batch.get("modalities", [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]

        if modalities:
            assert pred_camera_encs.shape[1] == len(modalities), (
                "Number of predicted camera encodings must match number of modalities."
            )

        losses = {}
        per_modality_preds = [pred_camera_encs[:, i, ...] for i in range(pred_camera_encs.shape[1])]
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            for pred_camera_enc, modality in zip(per_modality_preds, modalities):
                gt_camera = self._get_gt_camera_encoding(data_batch, modality, pred_camera_enc.device)
                if gt_camera is None:
                    continue
                loss_output = loss_fn(pred_camera_enc, gt_camera)
                if isinstance(loss_output, dict):
                    for k, v in loss_output.items():
                        losses[f"{loss_name}_{modality}_{k}"] = (v, loss_weight)
                else:
                    losses[f"{loss_name}_{modality}"] = (loss_output, loss_weight)

        proj_losses = self._projection_losses(per_modality_preds, modalities, data_batch)
        losses.update(proj_losses)
        return losses

    def predict(self, x, num_iterations: int = 4):
        return self.forward(x, num_iterations)

    def _select_layers(self, x):
        if not isinstance(x, list):
            return x
        if self.last_n_layers > 0:
            x = x[-self.last_n_layers :]
        if self.feature_slice == "first_half":
            x = [xi[..., : xi.shape[-1] // 2] for xi in x]
        elif self.feature_slice == "second_half":
            x = [xi[..., xi.shape[-1] // 2 :] for xi in x]
        elif self.feature_slice != "full":
            raise ValueError(f"Unknown feature_slice: {self.feature_slice}")
        return torch.cat(x, dim=-1)

    def _projection_losses(self, pred_camera_enc_list, modalities, data_batch):
        losses = {}
        gt_keypoints = data_batch.get("gt_keypoints", None)
        if gt_keypoints is None:
            return losses

        if isinstance(gt_keypoints, np.ndarray):
            gt_keypoints = torch.from_numpy(gt_keypoints)
        if not isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = torch.as_tensor(gt_keypoints, dtype=torch.float32)

        device = pred_camera_enc_list[0].device
        gt_keypoints = gt_keypoints.to(device).float()
        if gt_keypoints.dim() == 2:
            gt_keypoints = gt_keypoints.unsqueeze(0)

        for pred_camera_enc, modality in zip(pred_camera_enc_list, modalities):
            modality = modality.lower()
            if modality in {"rgb", "depth"} and self.proj_loss_weight_rgb > 0:
                image_size = self._get_image_size(data_batch, modality)
                gt_camera = self._get_gt_camera_encoding(data_batch, modality, device)
                if gt_camera is None:
                    continue
                pred_extrinsics, _ = self._pose_enc_to_extrinsics_intrinsics(
                    pred_camera_enc, None
                )
                gt_extrinsics, gt_intrinsics = self._pose_enc_to_extrinsics_intrinsics(
                    gt_camera, image_size
                )
                pred_proj = self._project_to_image(gt_keypoints, pred_extrinsics, gt_intrinsics)
                gt_proj = self._project_to_image(gt_keypoints, gt_extrinsics, gt_intrinsics)
                pred_proj = self._normalize_2d(pred_proj, image_size)
                gt_proj = self._normalize_2d(gt_proj, image_size)
                pred_proj = torch.clamp(pred_proj, -1.0, 1.0)
                gt_proj = torch.clamp(gt_proj, -1.0, 1.0)
                loss_val = self._projection_loss(pred_proj, gt_proj, self.proj_loss_type)
                losses[f"proj_{modality}"] = (loss_val, self.proj_loss_weight_rgb)
            elif modality in {"lidar", "mmwave"} and self.proj_loss_weight_lidar > 0:
                pred_extrinsics, _ = self._pose_enc_to_extrinsics_intrinsics(pred_camera_enc, None)
                gt_camera = self._get_gt_camera_encoding(data_batch, modality, device)
                if gt_camera is None:
                    continue
                gt_extrinsics, _ = self._pose_enc_to_extrinsics_intrinsics(gt_camera, None)
                pred_points = self._transform_to_camera(gt_keypoints, pred_extrinsics)
                gt_points = self._transform_to_camera(gt_keypoints, gt_extrinsics)
                loss_val = self._projection_loss(pred_points, gt_points, self.proj_loss_type)
                losses[f"proj_{modality}"] = (loss_val, self.proj_loss_weight_lidar)

        return losses

    def _pose_enc_to_extrinsics_intrinsics(self, pose_enc, image_size_hw):
        pose_enc = pose_enc.unsqueeze(1)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=image_size_hw,
            pose_encoding_type=self.pose_encoding_type,
            build_intrinsics=image_size_hw is not None,
        )
        return extrinsics.squeeze(1), None if intrinsics is None else intrinsics.squeeze(1)

    def _get_gt_camera_encoding(self, data_batch, modality, device):
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
        return gt_camera

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
        cam_points = VGGTCameraHeadV5._transform_to_camera(points, extrinsics)
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

    @staticmethod
    def _projection_loss(pred, target, loss_type="l1"):
        if loss_type.lower() in {"l1", "mae"}:
            loss = F.l1_loss(pred, target)
            return VGGTCameraHeadV5._sanitize_loss(loss)
        if loss_type.lower() in {"l2", "mse"}:
            loss = F.mse_loss(pred, target)
            return VGGTCameraHeadV5._sanitize_loss(loss)
        raise ValueError(f"Unsupported projection loss type: {loss_type}")

    @staticmethod
    def _sanitize_loss(loss):
        if loss is None:
            return loss
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros_like(loss)
        return loss
