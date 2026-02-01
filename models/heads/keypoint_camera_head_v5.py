import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_head import BaseHead
from misc.pose_enc import pose_encoding_to_extri_intri


class KeypointCameraHeadV5(BaseHead):
    """Predict camera pose encodings from keypoint predictions (global + per-modality)."""

    def __init__(
        self,
        losses,
        num_joints: int = 24,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        use_modality_embedding: bool = True,
        use_gt_ratio: float = 0.0,
        modalities_key: str = "modalities",
        pose_encoding_dim: int = 9,
        pose_encoding_type: str = "absT_quaR_FoV",
        proj_loss_weight_rgb: float = 1.0,
        proj_loss_weight_lidar: float = 1.0,
        proj_loss_type: str = "l1",
        detach_inputs: bool = True,
    ):
        super().__init__(losses)
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.use_modality_embedding = use_modality_embedding
        self.use_gt_ratio = float(use_gt_ratio)
        self.modalities_key = modalities_key
        self.pose_encoding_dim = pose_encoding_dim
        self.pose_encoding_type = pose_encoding_type
        self.proj_loss_weight_rgb = proj_loss_weight_rgb
        self.proj_loss_weight_lidar = proj_loss_weight_lidar
        self.proj_loss_type = proj_loss_type
        self.detach_inputs = detach_inputs

        input_dim = num_joints * 6
        if use_modality_embedding:
            self.modality_embed = nn.Embedding(8, hidden_dim)
            input_dim = input_dim + hidden_dim
        else:
            self.modality_embed = None

        layers = []
        in_dim = input_dim
        for idx in range(max(1, num_layers)):
            out_dim = hidden_dim if idx < num_layers - 1 else pose_encoding_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < num_layers - 1:
                if use_layernorm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, data_batch=None, pred_dict=None):
        return self.predict(x, data_batch=data_batch, pred_dict=pred_dict)

    def loss(self, x, data_batch, pred_dict=None):
        pred_encodings = self.predict(x, data_batch=data_batch, pred_dict=pred_dict)
        if pred_encodings is None:
            return {}

        modalities = self._get_modalities(data_batch)
        if not modalities:
            return {}

        losses = {}
        for m_idx, modality in enumerate(modalities):
            gt_key = f"gt_camera_{modality.lower()}"
            gt_camera = data_batch.get(gt_key)
            if gt_camera is None:
                continue
            gt_camera = self._to_tensor(gt_camera).to(pred_encodings.device)
            if gt_camera.dim() == 3:
                gt_camera = gt_camera[:, -1]
            if gt_camera.dim() == 2:
                gt_camera = gt_camera.unsqueeze(0) if gt_camera.shape[0] != pred_encodings.shape[0] else gt_camera

            pred = pred_encodings[:, m_idx]
            valid = torch.isfinite(pred).all(dim=-1)
            if valid.sum().item() == 0:
                continue
            pred = pred[valid]
            gt = gt_camera[valid]

            for loss_name, (loss_fn, loss_weight) in self.losses.items():
                loss_output = loss_fn(pred, gt)
                if isinstance(loss_output, dict):
                    for k, v in loss_output.items():
                        losses[f"{loss_name}_{modality}_{k}"] = (v, loss_weight)
                else:
                    losses[f"{loss_name}_{modality}"] = (loss_output, loss_weight)
        per_modality_preds = [pred_encodings[:, i, ...] for i in range(pred_encodings.shape[1])]
        proj_losses = self._projection_losses(per_modality_preds, modalities, data_batch)
        losses.update(proj_losses)
        return losses

    def predict(self, x, data_batch=None, pred_dict=None):
        if data_batch is None:
            return None

        modalities = self._get_modalities(data_batch)
        if not modalities:
            return None

        global_kps = self._get_global_keypoints(data_batch, pred_dict)
        if global_kps is None:
            return None

        global_kps = self._maybe_detach(global_kps)
        device = global_kps.device
        dtype = global_kps.dtype
        batch_size = global_kps.shape[0] if global_kps.dim() >= 3 else 1
        global_kps = self._ensure_batch(global_kps, batch_size)
        global_flat = global_kps.reshape(global_kps.shape[0], -1)
        num_modalities = len(modalities)

        pred_encodings = torch.full(
            (batch_size, num_modalities, self.pose_encoding_dim),
            float("nan"),
            device=device,
            dtype=dtype,
        )

        for m_idx, modality in enumerate(modalities):
            modality_l = modality.lower()
            if modality_l in {"rgb", "depth"}:
                kps = self._get_keypoints_2d(data_batch, pred_dict, modality_l, device, dtype)
            else:
                kps = self._get_keypoints_3d(data_batch, pred_dict, modality_l, device, dtype)
            if kps is None:
                continue

            kps = self._maybe_detach(kps)
            kps = self._ensure_batch(kps, batch_size)
            kps_3d = self._pad_2d_to_3d(kps)
            kps_flat = kps_3d.reshape(batch_size, -1)

            feat = torch.cat([global_flat, kps_flat], dim=-1)
            if self.use_modality_embedding and self.modality_embed is not None:
                mod_id = torch.full((batch_size,), m_idx, device=device, dtype=torch.long)
                feat = torch.cat([feat, self.modality_embed(mod_id)], dim=-1)

            pred_encodings[:, m_idx] = self.mlp(feat)

        return pred_encodings

    def _get_modalities(self, data_batch):
        modalities = data_batch.get(self.modalities_key, [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]
        return modalities

    def _get_global_keypoints(self, data_batch, pred_dict):
        pred = None
        if pred_dict is not None:
            pred = pred_dict.get("pred_keypoints")
        gt = data_batch.get("gt_keypoints")
        keypoints = self._mix_pred_gt(pred, gt)
        if keypoints is None:
            return None
        keypoints = self._to_tensor(keypoints)
        keypoints = self._select_frame(keypoints)
        return keypoints

    def _get_keypoints_2d(self, data_batch, pred_dict, modality, device, dtype):
        pred = None
        if pred_dict is not None:
            pred = pred_dict.get(f"pred_keypoints_2d_{modality}")
        gt = data_batch.get(f"gt_keypoints_2d_{modality}")
        keypoints = self._mix_pred_gt(pred, gt)
        if keypoints is None:
            return None
        keypoints = self._to_tensor(keypoints).to(device=device, dtype=dtype)
        keypoints = self._select_frame(keypoints)
        return keypoints

    def _get_keypoints_3d(self, data_batch, pred_dict, modality, device, dtype):
        pred = None
        if pred_dict is not None:
            if modality == "lidar":
                pred = pred_dict.get("pred_keypoints_pc_centered_input_lidar")
            if pred is None:
                pred = pred_dict.get(f"pred_keypoints_3d_{modality}")
        gt = data_batch.get(f"gt_keypoints_{modality}")
        if gt is None and modality == "lidar":
            gt = data_batch.get("gt_keypoints_pc_centered_input_lidar")
        keypoints = self._mix_pred_gt(pred, gt)
        if keypoints is None:
            return None
        keypoints = self._to_tensor(keypoints).to(device=device, dtype=dtype)
        keypoints = self._select_frame(keypoints)
        return keypoints

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, dtype=torch.float32)

    @staticmethod
    def _select_frame(x):
        if not isinstance(x, torch.Tensor):
            return x
        if x.dim() == 4:
            return x[:, -1]
        return x

    @staticmethod
    def _pad_2d_to_3d(points):
        if points.shape[-1] == 3:
            return points
        if points.shape[-1] != 2:
            return points
        zeros = torch.zeros(points.shape[:-1] + (1,), device=points.device, dtype=points.dtype)
        return torch.cat([points, zeros], dim=-1)

    def _maybe_detach(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        return tensor.detach() if self.detach_inputs else tensor

    @staticmethod
    def _ensure_batch(tensor, batch_size):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, *tensor.shape[1:])
        return tensor

    def _mix_pred_gt(self, pred, gt):
        pred = self._coerce_sequence(pred)
        gt = self._coerce_sequence(gt)
        if pred is None and gt is None:
            return None
        if pred is None:
            return gt
        if gt is None:
            return pred
        if not self.training or self.use_gt_ratio <= 0:
            return pred
        if self.use_gt_ratio >= 1.0:
            return gt
        if torch.rand(1).item() < self.use_gt_ratio:
            return gt
        return pred

    @staticmethod
    def _coerce_sequence(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            for v in value:
                if v is not None:
                    return v
            return None
        return value

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
                pred_extrinsics, _ = self._pose_enc_to_extrinsics_intrinsics(pred_camera_enc, None)
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
        cam_points = KeypointCameraHeadV5._transform_to_camera(points, extrinsics)
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
            return KeypointCameraHeadV5._sanitize_loss(loss)
        if loss_type.lower() in {"l2", "mse"}:
            loss = F.mse_loss(pred, target)
            return KeypointCameraHeadV5._sanitize_loss(loss)
        raise ValueError(f"Unsupported projection loss type: {loss_type}")

    @staticmethod
    def _sanitize_loss(loss):
        if loss is None:
            return loss
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros_like(loss)
        return loss
