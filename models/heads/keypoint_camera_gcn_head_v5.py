import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .base_head import BaseHead
from misc.pose_enc import pose_encoding_to_extri_intri
from models.aggregators.layers.gcn import TCN_GCN_unit
from misc.skeleton import (
    get_adjacency_matrix,
    COCOSkeleton,
    SimpleCOCOSkeleton,
    MMBodySkeleton,
    H36MSkeleton,
    MiliPointSkeleton,
    SMPLSkeleton,
)


class KeypointCameraGCNHeadV5(BaseHead):
    """Predict camera pose encodings from keypoints using separate 2D/3D ST-GCNs."""

    def __init__(
        self,
        losses,
        num_joints: int = 24,
        hidden_dim: int = 256,
        num_layers: int = 2,
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
        num_iterations: int = 1,
        gcn_kernel_size: int = 5,
        gcn_dilations: tuple = (1, 2),
        gcn_depth: int = 2,
        input_2d_skeleton_format: str = "coco",
        input_3d_skeleton_format: str = "smpl",
        train_branch: str = "both",
    ):
        super().__init__(losses)
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
        self.num_iterations = int(max(1, num_iterations))
        self.gcn_depth = int(max(1, gcn_depth))
        self.input_2d_skeleton_format = str(input_2d_skeleton_format).lower()
        self.input_3d_skeleton_format = str(input_3d_skeleton_format).lower()
        self.train_branch = str(train_branch).lower()
        if self.train_branch not in {"both", "2d", "3d"}:
            raise ValueError(
                f"Unsupported train_branch: {train_branch}. Expected one of ['both', '2d', '3d']."
            )

        self.target_skeleton = SimpleCOCOSkeleton()
        self.num_joints = self.target_skeleton.num_joints
        if num_joints != self.num_joints:
            warnings.warn(
                f"KeypointCameraGCNHeadV5 uses SimpleCOCO skeleton ({self.num_joints} joints). "
                f"Ignoring configured num_joints={num_joints}."
            )
        A = get_adjacency_matrix(self.target_skeleton.bones, self.target_skeleton.num_joints)

        self.gcn_in_channels = 9
        self.gcn_proj_2d = nn.Conv2d(5, self.gcn_in_channels, kernel_size=1)
        self.gcn_proj_3d = nn.Conv2d(6, self.gcn_in_channels, kernel_size=1)
        self.gcn_2d = self._build_gcn_stack(
            in_channels=self.gcn_in_channels,
            out_channels=hidden_dim,
            A=A,
            gcn_depth=self.gcn_depth,
            gcn_kernel_size=gcn_kernel_size,
            gcn_dilations=gcn_dilations,
        )
        self.gcn_3d = self._build_gcn_stack(
            in_channels=self.gcn_in_channels,
            out_channels=hidden_dim,
            A=A,
            gcn_depth=self.gcn_depth,
            gcn_kernel_size=gcn_kernel_size,
            gcn_dilations=gcn_dilations,
        )

        if use_modality_embedding:
            self.modality_embed = nn.Embedding(8, hidden_dim)
            mlp_in_dim = hidden_dim * 2
        else:
            self.modality_embed = None
            mlp_in_dim = hidden_dim

        self.prev_proj_2d = nn.Linear(pose_encoding_dim, mlp_in_dim)
        self.prev_proj_3d = nn.Linear(pose_encoding_dim, mlp_in_dim)
        self.mlp_2d = self._build_mlp(mlp_in_dim)
        self.mlp_3d = self._build_mlp(mlp_in_dim)
        self.post_gcn_norm_2d = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.post_gcn_norm_3d = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

    def forward(self, x, data_batch=None, pred_dict=None):
        return self.predict(x, data_batch=data_batch, pred_dict=pred_dict)

    def loss(self, x, data_batch, pred_dict=None):
        num_iterations = data_batch.get("num_camera_iterations", self.num_iterations)
        pred_encodings = self.predict(
            x, data_batch=data_batch, pred_dict=pred_dict, num_iterations=num_iterations
        )
        if pred_encodings is None:
            return {}

        modalities = self._get_modalities(data_batch)
        if not modalities:
            return {}

        losses = {}
        for m_idx, modality in enumerate(modalities):
            if not self._is_branch_enabled(modality):
                continue
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

    def predict(self, x, data_batch=None, pred_dict=None, num_iterations=None):
        if data_batch is None:
            return None

        modalities = self._get_modalities(data_batch)
        if not modalities:
            return None

        global_kps = self._get_global_keypoints(data_batch, pred_dict)
        if global_kps is None:
            return None

        global_kps = self._maybe_detach(global_kps)
        global_kps = self._convert_to_target_skeleton(
            global_kps,
            source_format=self.input_3d_skeleton_format,
        )
        if global_kps is None:
            return None
        device = global_kps.device
        dtype = global_kps.dtype
        batch_size = global_kps.shape[0] if global_kps.dim() >= 3 else 1
        global_kps = self._ensure_batch(global_kps, batch_size)
        num_modalities = len(modalities)

        pred_encodings = torch.full(
            (batch_size, num_modalities, self.pose_encoding_dim),
            float("nan"),
            device=device,
            dtype=dtype,
        )

        if num_iterations is None:
            num_iterations = self.num_iterations
        num_iterations = int(max(1, num_iterations))

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
            if modality_l in {"rgb", "depth"}:
                kps = self._convert_to_target_skeleton(
                    kps,
                    source_format=self.input_2d_skeleton_format,
                )
            else:
                kps = self._convert_to_target_skeleton(
                    kps,
                    source_format=self.input_3d_skeleton_format,
                )
            if kps is None:
                continue
            if modality_l in {"rgb", "depth"}:
                global_3d = self._pad_2d_to_3d(global_kps)
                if global_3d.shape[-1] != 3:
                    continue
                kps_2d = kps[..., :2] if kps.shape[-1] >= 2 else kps
                if kps_2d.shape[-1] != 2:
                    continue
                feat = torch.cat([global_3d, kps_2d], dim=-1)
                gcn_in = feat.permute(0, 2, 1).unsqueeze(2)
                gcn_in = self.gcn_proj_2d(gcn_in)
                gcn_out = self.gcn_2d(gcn_in)
                prev_proj = self.prev_proj_2d
                mlp = self.mlp_2d
                pooled = self.post_gcn_norm_2d(gcn_out.mean(dim=-1).mean(dim=-1))
            else:
                global_3d = self._pad_2d_to_3d(global_kps)
                if global_3d.shape[-1] != 3:
                    continue
                kps_3d = self._pad_2d_to_3d(kps)
                if kps_3d.shape[-1] != 3:
                    continue
                feat = torch.cat([global_3d, kps_3d], dim=-1)
                gcn_in = feat.permute(0, 2, 1).unsqueeze(2)
                gcn_in = self.gcn_proj_3d(gcn_in)
                gcn_out = self.gcn_3d(gcn_in)
                prev_proj = self.prev_proj_3d
                mlp = self.mlp_3d
                pooled = self.post_gcn_norm_3d(gcn_out.mean(dim=-1).mean(dim=-1))
            if self.use_modality_embedding and self.modality_embed is not None:
                mod_id = torch.full((batch_size,), m_idx, device=device, dtype=torch.long)
                pooled = torch.cat([pooled, self.modality_embed(mod_id)], dim=-1)

            pred = None
            for _ in range(num_iterations):
                if pred is not None:
                    pred = pred.detach()
                    feat_iter = pooled + prev_proj(pred)
                else:
                    feat_iter = pooled
                delta = mlp(feat_iter)
                pred = delta if pred is None else pred + delta

            pred_encodings[:, m_idx] = pred

        return pred_encodings

    def _build_mlp(self, input_dim):
        layers = []
        in_dim = input_dim
        for idx in range(max(1, self.num_layers)):
            out_dim = self.hidden_dim if idx < self.num_layers - 1 else self.pose_encoding_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < self.num_layers - 1:
                if self.use_layernorm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        return nn.Sequential(*layers)

    def _build_gcn_stack(self, in_channels, out_channels, A, gcn_depth, gcn_kernel_size, gcn_dilations):
        layers = []
        for idx in range(gcn_depth):
            layers.append(
                TCN_GCN_unit(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    A=A,
                    residual=idx != 0,
                    kernel_size=gcn_kernel_size,
                    dilations=list(gcn_dilations),
                )
            )
        return nn.Sequential(*layers)


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
        if modality == "rgb":
            # Prefer dataset-provided RGB 2D skeletons (e.g., JSON-loaded),
            # and only fall back to predicted RGB keypoints if unavailable.
            keypoints = self._coerce_sequence(gt)
            if keypoints is None:
                keypoints = self._coerce_sequence(pred)
        else:
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

    @staticmethod
    def _build_source_skeleton(source_format: str):
        sf = str(source_format).lower()
        if sf in {"smpl"}:
            return SMPLSkeleton
        if sf in {"mmbody"}:
            return MMBodySkeleton
        if sf in {"h36m", "mmfi"}:
            return H36MSkeleton
        if sf in {"coco"}:
            return COCOSkeleton
        if sf in {"simple_coco", "simplecoco"}:
            return SimpleCOCOSkeleton
        if sf in {"milipoint"}:
            return MiliPointSkeleton
        return None

    def _resolve_source_format(self, points, source_format: str) -> str:
        sf = str(source_format).lower()
        if sf == "simplecoco":
            sf = "simple_coco"
        if sf != "auto":
            return sf
        j = int(points.shape[-2])
        if j == SMPLSkeleton.num_joints:
            return "smpl"
        if j == MMBodySkeleton.num_joints:
            return "mmbody"
        if j == H36MSkeleton.num_joints:
            return "h36m"
        if j == COCOSkeleton.num_joints:
            return "coco"
        if j == SimpleCOCOSkeleton.num_joints:
            return "simple_coco"
        if j == MiliPointSkeleton.num_joints:
            return "milipoint"
        return "unknown"

    def _convert_to_target_skeleton(self, points, source_format: str):
        if points is None:
            return None
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points, dtype=torch.float32)
        if points.dim() < 2:
            return None

        sf = self._resolve_source_format(points, source_format)
        source_skeleton = self._build_source_skeleton(sf)
        if source_skeleton is None:
            if points.shape[-2] >= self.num_joints:
                return points[..., : self.num_joints, :]
            pad_n = self.num_joints - points.shape[-2]
            pad = torch.zeros(*points.shape[:-2], pad_n, points.shape[-1], device=points.device, dtype=points.dtype)
            return torch.cat([points, pad], dim=-2)

        if not hasattr(source_skeleton, "to_simple_coco"):
            raise ValueError(
                f"Skeleton format '{sf}' does not provide to_simple_coco conversion."
            )

        converted = source_skeleton.to_simple_coco(points)
        if converted.shape[-2] > self.num_joints:
            converted = converted[..., : self.num_joints, :]
        elif converted.shape[-2] < self.num_joints:
            pad_n = self.num_joints - converted.shape[-2]
            pad = torch.zeros(
                *converted.shape[:-2], pad_n, converted.shape[-1],
                device=converted.device, dtype=converted.dtype
            )
            converted = torch.cat([converted, pad], dim=-2)
        return converted

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
            if not self._is_branch_enabled(modality):
                continue
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

    def _is_branch_enabled(self, modality: str) -> bool:
        modality_l = str(modality).lower()
        is_2d = modality_l in {"rgb", "depth"}
        if self.train_branch == "both":
            return True
        if self.train_branch == "2d":
            return is_2d
        return not is_2d

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
        size_key = f"image_size_hw_{modality}"
        if size_key in data_batch:
            size = data_batch[size_key]
            if isinstance(size, (list, tuple)) and len(size) == 2 and not isinstance(size[0], (list, tuple)):
                return int(size[0]), int(size[1])
            if isinstance(size, (list, tuple)) and len(size) > 0 and isinstance(size[0], (list, tuple)) and len(size[0]) == 2:
                return int(size[0][0]), int(size[0][1])
        if "image_size_hw" in data_batch:
            size = data_batch["image_size_hw"]
            if isinstance(size, (list, tuple)) and len(size) == 2 and not isinstance(size[0], (list, tuple)):
                return int(size[0]), int(size[1])
            if isinstance(size, (list, tuple)) and len(size) > 0 and isinstance(size[0], (list, tuple)) and len(size[0]) == 2:
                return int(size[0][0]), int(size[0][1])
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
        cam_points = KeypointCameraGCNHeadV5._transform_to_camera(points, extrinsics)
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
            return KeypointCameraGCNHeadV5._sanitize_loss(loss)
        if loss_type.lower() in {"l2", "mse"}:
            loss = F.mse_loss(pred, target)
            return KeypointCameraGCNHeadV5._sanitize_loss(loss)
        raise ValueError(f"Unsupported projection loss type: {loss_type}")

    @staticmethod
    def _sanitize_loss(loss):
        if loss is None:
            return loss
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros_like(loss)
        return loss
