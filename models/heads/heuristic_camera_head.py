import math
import torch
import numpy as np
from typing import Iterable, Optional, Tuple

from .base_head import BaseHead
from misc.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from misc.camera_batch import get_gt_camera_encoding


class HeuristicCameraHead(BaseHead):
    """Estimate per-modality camera pose encodings from keypoint correspondences.

    Uses:
    - RGB/Depth: PnP (DLT) with known intrinsics and 2D keypoints.
    - LiDAR: rigid 3D-3D alignment against global 3D keypoints.
    """

    def __init__(
        self,
        losses,
        modalities_key: str = "modalities",
        global_keypoints_key: str = "pred_keypoints",
        global_fallback_key: str = "gt_keypoints",
        pred_keypoints_2d_key_templates: Tuple[str, ...] = (
            "pred_keypoints_2d_{modality}",
        ),
        pred_keypoints_3d_key_templates: Tuple[str, ...] = (
            "pred_keypoints_pc_centered_input_{modality}",
            "pred_keypoints_3d_{modality}",
        ),
        keypoints_2d_key_templates: Tuple[str, ...] = (
            "keypoints_2d_{modality}",
            "gt_keypoints_2d_{modality}",
        ),
        keypoints_3d_key_templates: Tuple[str, ...] = (
            "keypoints_3d_{modality}",
            "gt_keypoints_{modality}",
            "gt_keypoints_pc_centered_input_{modality}",
        ),
        intrinsics_key_templates: Tuple[str, ...] = (
            "intrinsics_{modality}",
            "camera_intrinsics_{modality}",
            "K_{modality}",
        ),
        intrinsics: Optional[Iterable[Iterable[float]]] = None,
        keypoints_2d_normalized: bool = False,
        default_image_size: Tuple[int, int] = (224, 224),
        default_fov_deg: Optional[Tuple[float, float]] = None,
        pose_encoding_type: str = "absT_quaR_FoV",
        estimate_lidar_scale: bool = False,
        min_points_pnp: int = 6,
        min_points_3d: int = 3,
    ):
        super().__init__(losses)
        self.modalities_key = modalities_key
        self.global_keypoints_key = global_keypoints_key
        self.global_fallback_key = global_fallback_key
        self.pred_keypoints_2d_key_templates = pred_keypoints_2d_key_templates
        self.pred_keypoints_3d_key_templates = pred_keypoints_3d_key_templates
        self.keypoints_2d_key_templates = keypoints_2d_key_templates
        self.keypoints_3d_key_templates = keypoints_3d_key_templates
        self.intrinsics_key_templates = intrinsics_key_templates
        self.keypoints_2d_normalized = keypoints_2d_normalized
        self.default_image_size = default_image_size
        self.default_fov_deg = default_fov_deg
        self.pose_encoding_type = pose_encoding_type
        self.estimate_lidar_scale = estimate_lidar_scale
        self.min_points_pnp = min_points_pnp
        self.min_points_3d = min_points_3d

        if intrinsics is not None:
            self.register_buffer("_fixed_intrinsics", torch.as_tensor(intrinsics, dtype=torch.float32))
        else:
            self._fixed_intrinsics = None

    def forward(self, x, data_batch=None, pred_dict=None):
        return self.predict(x, data_batch=data_batch, pred_dict=pred_dict)

    def loss(self, x, data_batch):
        return {}

    def predict(self, x, data_batch=None, pred_dict=None):
        if data_batch is None:
            return None

        modalities = self._get_modalities(data_batch)
        if not modalities:
            return None

        global_kps = self._get_global_keypoints(data_batch, pred_dict)
        if global_kps is None:
            return None

        device = global_kps.device
        dtype = global_kps.dtype
        batch_size = global_kps.shape[0]
        num_modalities = len(modalities)

        pred_extrinsics = torch.full(
            (batch_size, num_modalities, 4, 4),
            float("nan"),
            device=device,
            dtype=dtype,
        )
        pred_encodings = torch.full(
            (batch_size, num_modalities, 9),
            float("nan"),
            device=device,
            dtype=dtype,
        )

        for m_idx, modality in enumerate(modalities):
            modality = modality.lower()
            if modality in {"rgb", "depth"}:
                keypoints_2d = self._get_keypoints_2d(
                    data_batch, pred_dict, modality, device, dtype, batch_size
                )
                intrinsics = self._get_intrinsics(data_batch, modality, device, dtype, batch_size)
                if keypoints_2d is None or intrinsics is None:
                    continue
                intrinsics = self._expand_intrinsics(intrinsics, batch_size)
                if self.keypoints_2d_normalized:
                    image_size = self._get_image_size(data_batch, modality)
                    keypoints_2d = keypoints_2d.clamp(-1.0, 1.0)
                    keypoints_2d = self._denormalize_2d(keypoints_2d, image_size)
                pred_extrinsics[:, m_idx] = self._solve_pnp_batch(
                    global_kps, keypoints_2d, intrinsics
                )
                pred_encodings[:, m_idx] = self._encode_pose(
                    pred_extrinsics[:, m_idx], intrinsics, image_size=self._get_image_size(data_batch, modality)
                )
            elif modality == "lidar":
                keypoints_3d = self._get_keypoints_3d(
                    data_batch, pred_dict, modality, device, dtype, batch_size
                )
                if keypoints_3d is None:
                    continue
                pred_extrinsics[:, m_idx] = self._solve_rigid_batch(
                    global_kps, keypoints_3d, estimate_scale=self.estimate_lidar_scale
                )
                intrinsics = self._get_intrinsics(data_batch, modality, device, dtype, batch_size)
                if intrinsics is None and self.default_fov_deg is not None:
                    intrinsics = self._intrinsics_from_fov(
                        self._get_image_size(data_batch, modality),
                        self.default_fov_deg,
                        device=device,
                        dtype=dtype,
                        batch_size=batch_size,
                    )
                if intrinsics is not None:
                    intrinsics = self._expand_intrinsics(intrinsics, batch_size)
                pred_encodings[:, m_idx] = self._encode_pose(
                    pred_extrinsics[:, m_idx], intrinsics, image_size=self._get_image_size(data_batch, modality)
                )

        return pred_encodings

    def _get_modalities(self, data_batch):
        modalities = data_batch.get(self.modalities_key, [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]
        return modalities

    def _get_global_keypoints(self, data_batch, pred_dict):
        keypoints = None
        if pred_dict is not None:
            keypoints = pred_dict.get(self.global_keypoints_key, None)
        if keypoints is None:
            keypoints = data_batch.get(self.global_keypoints_key, None)
        if keypoints is None:
            keypoints = data_batch.get(self.global_fallback_key, None)
        if keypoints is None:
            return None

        keypoints = self._to_tensor(keypoints)
        if keypoints is None:
            return None
        batch_size = self._infer_batch_size(data_batch, keypoints)
        keypoints = self._ensure_bstjc(keypoints, batch_size)
        if keypoints is None:
            return None
        keypoints = keypoints[:, -1]
        return keypoints

    def _get_keypoints_2d(self, data_batch, pred_dict, modality, device, dtype, batch_size):
        pred = self._get_by_templates(pred_dict, modality, self.pred_keypoints_2d_key_templates)
        if pred is not None:
            pred = self._to_tensor(pred).to(device=device, dtype=dtype)
            pred = self._ensure_bstjc(pred, batch_size)
            if pred is not None:
                pred = pred[:, -1]
                valid = torch.isfinite(pred).all(dim=-1)
                if valid.sum().item() >= self.min_points_pnp:
                    return pred
        keypoints = self._get_by_templates(data_batch, modality, self.keypoints_2d_key_templates)
        if keypoints is None:
            return None
        keypoints = self._to_tensor(keypoints).to(device=device, dtype=dtype)
        keypoints = self._ensure_bstjc(keypoints, batch_size)
        if keypoints is None:
            return None
        keypoints = keypoints[:, -1]
        return keypoints

    def _get_keypoints_3d(self, data_batch, pred_dict, modality, device, dtype, batch_size):
        pred = self._get_by_templates(pred_dict, modality, self.pred_keypoints_3d_key_templates)
        if pred is not None:
            pred = self._to_tensor(pred).to(device=device, dtype=dtype)
            pred = self._ensure_bstjc(pred, batch_size)
            if pred is not None:
                pred = pred[:, -1]
                valid = torch.isfinite(pred).all(dim=-1)
                if valid.sum().item() >= self.min_points_3d:
                    return pred
        keypoints = self._get_by_templates(data_batch, modality, self.keypoints_3d_key_templates)
        if keypoints is None:
            return None
        keypoints = self._to_tensor(keypoints).to(device=device, dtype=dtype)
        keypoints = self._ensure_bstjc(keypoints, batch_size)
        if keypoints is None:
            return None
        keypoints = keypoints[:, -1]
        return keypoints

    def _get_intrinsics(self, data_batch, modality, device, dtype, batch_size):
        if self._fixed_intrinsics is not None:
            intrinsics = self._fixed_intrinsics
        else:
            intrinsics = self._get_by_templates(data_batch, modality, self.intrinsics_key_templates)
        if intrinsics is None:
            intrinsics = self._get_intrinsics_from_camera_dict(data_batch, modality)
        if intrinsics is None:
            intrinsics = self._get_intrinsics_from_pose_encoding(data_batch, modality)
        if intrinsics is None:
            return None
        intrinsics = self._to_tensor(intrinsics).to(device=device, dtype=dtype)
        return self._normalize_intrinsics_shape(intrinsics, batch_size)

    def _get_by_templates(self, data_batch, modality, templates):
        for template in templates:
            key = template.format(modality=modality)
            if key in data_batch:
                return data_batch[key]
        return None

    def _get_image_size(self, data_batch, modality):
        input_key = f"input_{modality}"
        if input_key in data_batch:
            inp = data_batch[input_key]
            if isinstance(inp, torch.Tensor) and inp.dim() >= 4:
                return int(inp.shape[-2]), int(inp.shape[-1])
        return self.default_image_size

    def _get_intrinsics_from_camera_dict(self, data_batch, modality):
        cam_key = f"{modality}_camera"
        camera = data_batch.get(cam_key)
        if camera is None:
            return None
        if isinstance(camera, dict):
            return camera.get("intrinsic")
        if not isinstance(camera, (list, tuple)):
            return None
        intrinsics = []
        for cam in camera:
            if cam is None or not isinstance(cam, dict):
                intrinsics.append(None)
                continue
            intrinsics.append(cam.get("intrinsic"))
        if all(k is None for k in intrinsics):
            return None
        tensors = []
        for k in intrinsics:
            if k is None:
                tensors.append(torch.full((3, 3), float("nan")))
            else:
                tensors.append(torch.as_tensor(k, dtype=torch.float32))
        return torch.stack(tensors, dim=0)

    def _get_intrinsics_from_pose_encoding(self, data_batch, modality):
        batch_size = self._infer_batch_size(data_batch, None)
        pose_enc = get_gt_camera_encoding(
            data_batch=data_batch,
            modality=modality,
            batch_size=batch_size,
            device=torch.device("cpu"),
            dtype=torch.float32,
            pose_encoding_dim=9,
        )
        if pose_enc is None:
            return None
        pose_enc = pose_enc.unsqueeze(1)
        _, intrinsics = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=self._get_image_size(data_batch, modality),
            pose_encoding_type=self.pose_encoding_type,
            build_intrinsics=True,
        )
        return intrinsics.squeeze(1)

    @staticmethod
    def _to_tensor(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return None
            items = []
            ref = None
            for item in x:
                t = HeuristicCameraHead._to_tensor(item)
                if t is None:
                    items.append(None)
                    continue
                t = t.float()
                if ref is None:
                    ref = t
                items.append(t)
            if ref is None:
                return None
            ref_shape = ref.shape
            ref_device = ref.device
            ref_dtype = ref.dtype
            stacked = []
            for t in items:
                if t is None:
                    stacked.append(torch.zeros(ref_shape, dtype=ref_dtype, device=ref_device))
                    continue
                if t.shape != ref_shape:
                    if t.numel() == ref.numel():
                        t = t.reshape(ref_shape)
                    else:
                        stacked.append(torch.zeros(ref_shape, dtype=ref_dtype, device=ref_device))
                        continue
                stacked.append(t.to(device=ref_device, dtype=ref_dtype))
            return torch.stack(stacked, dim=0)
        return torch.as_tensor(x, dtype=torch.float32)

    @staticmethod
    def _select_frame(x):
        if not isinstance(x, torch.Tensor):
            return x
        if x.dim() == 5:
            # [B, V, S, J, C] -> average over views, use last time step.
            return x.mean(dim=1)[:, -1]
        if x.dim() == 4:
            # [B, S, J, C] -> use last time step.
            return x[:, -1]
        return x

    @staticmethod
    def _ensure_bstjc(tensor, batch_size):
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.dim() == 5:
            if tensor.shape[0] == batch_size:
                tensor = tensor.mean(dim=1)
            elif tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.expand(batch_size, -1, -1, -1, -1).mean(dim=1)
            elif batch_size == 1:
                tensor = tensor.mean(dim=0, keepdim=True)
            else:
                return None
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            if tensor.shape[0] == batch_size:
                tensor = tensor.unsqueeze(1)
            elif tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.unsqueeze(1).expand(batch_size, -1, -1, -1)
            elif batch_size == 1:
                tensor = tensor.unsqueeze(0)
            else:
                return None
        elif tensor.dim() == 4:
            if tensor.shape[0] == batch_size:
                pass
            elif tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.expand(batch_size, -1, -1, -1)
            elif batch_size == 1:
                tensor = tensor.mean(dim=0, keepdim=True)
            else:
                return None
        else:
            return None
        return tensor

    @staticmethod
    def _normalize_intrinsics_shape(intrinsics, batch_size):
        if not isinstance(intrinsics, torch.Tensor):
            return None
        if intrinsics.dim() == 5:
            # [B, V, S, 3, 3] -> use last time step, mean over views.
            if intrinsics.shape[0] == batch_size:
                return intrinsics[:, :, -1].mean(dim=1)
            if intrinsics.shape[0] == 1 and batch_size > 1:
                intrinsics = intrinsics.expand(batch_size, -1, -1, -1, -1)
                return intrinsics[:, :, -1].mean(dim=1)
            if batch_size == 1:
                flat = intrinsics.reshape(-1, intrinsics.shape[-3], intrinsics.shape[-2], intrinsics.shape[-1])
                return flat[:, -1].mean(dim=0, keepdim=True)
            return None
        if intrinsics.dim() == 4:
            # [B, S, 3, 3] -> use last time step.
            if intrinsics.shape[0] == batch_size:
                return intrinsics[:, -1]
            if intrinsics.shape[0] == 1 and batch_size > 1:
                intrinsics = intrinsics.expand(batch_size, -1, -1, -1)
                return intrinsics[:, -1]
            if batch_size == 1:
                return intrinsics[:, -1].mean(dim=0, keepdim=True)
            return None
        if intrinsics.dim() == 3:
            if intrinsics.shape[0] == batch_size:
                return intrinsics
            if intrinsics.shape[0] == 1 and batch_size > 1:
                return intrinsics.expand(batch_size, -1, -1)
            if batch_size == 1:
                return intrinsics[-1:].contiguous()
            return None
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0)
            if batch_size > 1:
                intrinsics = intrinsics.expand(batch_size, -1, -1)
            return intrinsics
        return None

    def _infer_batch_size(self, data_batch, tensor_fallback=None):
        sample_ids = data_batch.get("sample_id", None)
        if isinstance(sample_ids, (list, tuple)):
            return len(sample_ids)
        if isinstance(sample_ids, torch.Tensor) and sample_ids.dim() > 0:
            return int(sample_ids.shape[0])
        if isinstance(tensor_fallback, torch.Tensor) and tensor_fallback.dim() >= 3:
            return int(tensor_fallback.shape[0])
        for key, value in data_batch.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)) and len(value) > 0 and (
                key == "sample_id" or str(key).startswith("input_") or str(key).startswith("gt_")
            ):
                return len(value)
        return 1

    @staticmethod
    def _denormalize_2d(points_2d, image_size_hw):
        height, width = image_size_hw
        x = (points_2d[..., 0] + 1.0) * 0.5 * (width - 1)
        y = (points_2d[..., 1] + 1.0) * 0.5 * (height - 1)
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _expand_intrinsics(intrinsics, batch_size):
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.shape[0] == 1 and batch_size > 1:
            intrinsics = intrinsics.expand(batch_size, -1, -1)
        return intrinsics

    def _encode_pose(self, extrinsics, intrinsics, image_size):
        if intrinsics is None:
            return torch.full(
                (extrinsics.shape[0], 9),
                float("nan"),
                device=extrinsics.device,
                dtype=extrinsics.dtype,
            )
        extrinsics_34 = extrinsics[:, :3, :4].unsqueeze(1)
        intrinsics = intrinsics.unsqueeze(1) if intrinsics.dim() == 3 else intrinsics
        pose = extri_intri_to_pose_encoding(
            extrinsics_34, intrinsics, image_size_hw=image_size, pose_encoding_type=self.pose_encoding_type
        )
        return pose.squeeze(1)

    @staticmethod
    def _intrinsics_from_fov(image_size_hw, fov_deg, device, dtype, batch_size):
        height, width = image_size_hw
        fov_h = math.radians(fov_deg[0])
        fov_w = math.radians(fov_deg[1])
        fy = (height / 2.0) / math.tan(fov_h / 2.0)
        fx = (width / 2.0) / math.tan(fov_w / 2.0)
        intrinsics = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = width / 2.0
        intrinsics[:, 1, 2] = height / 2.0
        intrinsics[:, 2, 2] = 1.0
        return intrinsics

    def _solve_pnp_batch(self, points_3d, points_2d, intrinsics):
        batch_size = points_3d.shape[0]
        extrinsics = []
        with torch.cuda.amp.autocast(enabled=False):
            for b in range(batch_size):
                X = points_3d[b]
                x = points_2d[b]
                if intrinsics.dim() == 3:
                    K = intrinsics[0] if intrinsics.shape[0] == 1 else intrinsics[b]
                else:
                    K = intrinsics

                orig_dtype = X.dtype
                X = X.float()
                x = x.float()
                K = K.float()

                valid = torch.isfinite(X).all(dim=-1) & torch.isfinite(x).all(dim=-1)
                X = X[valid]
                x = x[valid]

                if X.shape[0] < self.min_points_pnp:
                    extrinsics.append(self._nan_extrinsics(X.device, X.dtype))
                    continue

                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                u = (x[:, 0] - cx) / fx
                v = (x[:, 1] - cy) / fy

                ones = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
                Xh = torch.cat([X, ones], dim=1)
                zeros = torch.zeros_like(Xh)

                row1 = torch.cat([zeros, -Xh, v[:, None] * Xh], dim=1)
                row2 = torch.cat([Xh, zeros, -u[:, None] * Xh], dim=1)
                A = torch.cat([row1, row2], dim=0)

                _, _, Vh = torch.linalg.svd(A)
                P = Vh[-1].reshape(3, 4)

                R_tilde = P[:, :3]
                t_tilde = P[:, 3]
                U, S, Vh_r = torch.linalg.svd(R_tilde)
                R = U @ Vh_r
                if torch.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vh_r

                scale = S.mean()
                t = t_tilde / scale

                extrinsics.append(self._rt_to_matrix(R, t).to(orig_dtype))

        return torch.stack(extrinsics, dim=0)

    def _solve_rigid_batch(self, src_points, dst_points, estimate_scale=False):
        batch_size = src_points.shape[0]
        extrinsics = []
        with torch.cuda.amp.autocast(enabled=False):
            for b in range(batch_size):
                X = src_points[b]
                Y = dst_points[b]

                orig_dtype = X.dtype
                X = X.float()
                Y = Y.float()

                valid = torch.isfinite(X).all(dim=-1) & torch.isfinite(Y).all(dim=-1)
                X = X[valid]
                Y = Y[valid]

                if X.shape[0] < self.min_points_3d:
                    extrinsics.append(self._nan_extrinsics(X.device, X.dtype))
                    continue

                mu_x = X.mean(dim=0)
                mu_y = Y.mean(dim=0)
                Xc = X - mu_x
                Yc = Y - mu_y

                cov = Xc.t().mm(Yc) / X.shape[0]
                U, S, Vh = torch.linalg.svd(cov)
                R = Vh.t().mm(U.t())
                if torch.linalg.det(R) < 0:
                    Vh[-1, :] *= -1
                    R = Vh.t().mm(U.t())

                scale = 1.0
                if estimate_scale:
                    var_x = (Xc ** 2).sum() / X.shape[0]
                    if var_x > 0:
                        scale = S.sum() / var_x

                t = mu_y - scale * R.mm(mu_x.unsqueeze(1)).squeeze(1)

                extrinsics.append(self._rt_to_matrix(R * scale, t).to(orig_dtype))

        return torch.stack(extrinsics, dim=0)

    @staticmethod
    def _rt_to_matrix(R, t):
        extr = torch.eye(4, device=R.device, dtype=R.dtype)
        extr[:3, :3] = R
        extr[:3, 3] = t
        return extr

    @staticmethod
    def _nan_extrinsics(device, dtype):
        return torch.full((4, 4), float("nan"), device=device, dtype=dtype)
