import numpy as np
import torch

from misc.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


class CameraParamToPoseEncoding:
    """Convert camera intrinsics/extrinsics into pose encodings for supervision."""

    def __init__(self, pose_encoding_type: str = "absT_quaR_FoV"):
        self.pose_encoding_type = pose_encoding_type

    def __call__(self, sample):
        modalities = sample.get("modalities", [])

        for modality in modalities:
            camera_key = f"{modality}_camera"
            input_key = f"input_{modality}"

            camera = sample.get(camera_key)
            frames = sample.get(input_key)
            if camera is None or frames is None:
                continue

            # Prepare batched extrinsics/intrinsics for the sequence length
            seq_len = len(frames)
            extrinsic = np.asarray(camera["extrinsic"], dtype=np.float32)
            intrinsic = np.asarray(camera["intrinsic"], dtype=np.float32)

            extrinsics = torch.from_numpy(np.stack([extrinsic] * seq_len, axis=0)).unsqueeze(0)
            intrinsics = torch.from_numpy(np.stack([intrinsic] * seq_len, axis=0)).unsqueeze(0)

            # Image size is (H, W)
            height, width = frames[0].shape[:2]
            pose_enc = extri_intri_to_pose_encoding(
                extrinsics,
                intrinsics,
                image_size_hw=(height, width),
                pose_encoding_type=self.pose_encoding_type,
            )

            # Store per-sample (S x 9) pose encoding; collate will add batch dim
            sample[f"gt_camera_{modality}"] = pose_enc.squeeze(0)

        return sample


def _axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _axis_angle_to_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle)
    if angle.item() < 1e-8:
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    axis = axis_angle / angle
    x, y, z = axis
    K = torch.tensor(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        device=axis_angle.device,
        dtype=axis_angle.dtype,
    )
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return eye + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


class KeypointsToNewWorld:
    """Apply pelvis-centered, root-rotation-removed transform to gt_keypoints.

    Expects:
        gt_global_orient: axis-angle (3,)
        gt_pelvis: (3,)
    """

    def __init__(self, key: str = "gt_keypoints"):
        self.key = key

    def __call__(self, sample):
        global_orient = sample.get("gt_global_orient")
        pelvis = sample.get("gt_pelvis")
        if global_orient is None or pelvis is None:
            return sample

        if self.key not in sample:
            return sample

        if isinstance(global_orient, torch.Tensor):
            R_root = _axis_angle_to_matrix_torch(global_orient)
            pelvis_t = pelvis if isinstance(pelvis, torch.Tensor) else torch.as_tensor(
                pelvis, dtype=global_orient.dtype, device=global_orient.device
            )
            value = sample[self.key]
            if isinstance(value, torch.Tensor) and value.shape[-1] == 3:
                sample[self.key] = self._transform_torch(value, R_root, pelvis_t)
        else:
            R_root = _axis_angle_to_matrix_np(np.asarray(global_orient, dtype=np.float32))
            pelvis_np = np.asarray(pelvis, dtype=np.float32)
            value = sample[self.key]
            if isinstance(value, np.ndarray) and value.shape[-1] == 3:
                sample[self.key] = self._transform_np(value, R_root, pelvis_np)

        sample["gt_pelvis"] = np.zeros(3, dtype=np.float32)
        return sample

    @staticmethod
    def _transform_np(points, R_root, pelvis):
        shape = points.shape
        pts = points.reshape(-1, 3)
        pts = (R_root.T @ (pts - pelvis).T).T
        return pts.reshape(shape)

    @staticmethod
    def _transform_torch(points, R_root, pelvis):
        shape = points.shape
        pts = points.reshape(-1, 3)
        pts = (R_root.t() @ (pts - pelvis).t()).t()
        return pts.reshape(shape)


class ReplaceBySyntheticCamera:
    """Replace camera params with synthetic (perturbed) ones and update keypoints."""

    def __init__(
        self,
        synthetic_prob: float = 1.0,
        rot_jitter_deg: float = 5.0,
        trans_jitter: float = 0.05,
        focal_scale: tuple = (0.8, 1.2),
        modalities: tuple = None,
        normalize_2d: bool = True,
        default_image_size_hw: tuple = (224, 224),
    ):
        self.synthetic_prob = float(synthetic_prob)
        self.rot_jitter_deg = float(rot_jitter_deg)
        self.trans_jitter = float(trans_jitter)
        self.focal_scale = focal_scale
        self.modalities = modalities
        self.normalize_2d = bool(normalize_2d)
        self.default_image_size_hw = (int(default_image_size_hw[0]), int(default_image_size_hw[1]))

    def __call__(self, sample):
        if self.synthetic_prob <= 0:
            apply_synth = False
        else:
            apply_synth = np.random.rand() <= self.synthetic_prob

        modalities = self.modalities or sample.get("modalities", [])
        gt_keypoints = sample.get("gt_keypoints", None)

        for modality in modalities:
            camera_key = f"{modality}_camera"
            camera = sample.get(camera_key)
            if camera is None:
                continue
            intrinsic = np.asarray(camera["intrinsic"], dtype=np.float32)
            extrinsic = np.asarray(camera["extrinsic"], dtype=np.float32)

            if gt_keypoints is not None:
                self._update_keypoints(sample, modality, gt_keypoints, extrinsic, intrinsic)

            if not apply_synth:
                continue

            synthetic_cam = self._jitter_camera(intrinsic, extrinsic)
            sample[camera_key] = synthetic_cam

            if gt_keypoints is not None:
                self._update_keypoints(
                    sample,
                    modality,
                    gt_keypoints,
                    synthetic_cam["extrinsic"],
                    synthetic_cam["intrinsic"],
                )

        return sample

    def _jitter_camera(self, intrinsic, extrinsic):
        R = extrinsic[:, :3].copy()
        T = extrinsic[:, 3:].copy()

        if self.rot_jitter_deg > 0:
            axis = np.random.randn(3).astype(np.float32)
            axis /= np.linalg.norm(axis) + 1e-8
            angle = np.deg2rad(np.random.uniform(-self.rot_jitter_deg, self.rot_jitter_deg))
            K = np.array(
                [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
                dtype=np.float32,
            )
            R_jitter = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
            R = R_jitter @ R

        if self.trans_jitter > 0:
            T = T + np.random.randn(3, 1).astype(np.float32) * self.trans_jitter

        K = intrinsic.copy()
        if self.focal_scale is not None:
            scale = np.random.uniform(self.focal_scale[0], self.focal_scale[1])
            K[0, 0] *= scale
            K[1, 1] *= scale

        return {
            "intrinsic": K.astype(np.float32),
            "extrinsic": np.hstack((R, T)).astype(np.float32),
        }

    def _update_keypoints(self, sample, modality, gt_keypoints, extrinsic, intrinsic):
        if modality in {"rgb", "depth"}:
            image_size_hw = self._get_image_size(sample, modality)
            kp_2d = self._project_to_image(gt_keypoints, extrinsic, intrinsic)
            if self.normalize_2d:
                kp_2d = self._normalize_2d(kp_2d, image_size_hw)
                kp_2d = np.clip(kp_2d, -1.0, 1.0)
            sample[f"gt_keypoints_2d_{modality}"] = kp_2d.astype(np.float32)
            return

        # cam_kp_3d = self._transform_to_camera(gt_keypoints, extrinsic)
        # sample[f"gt_keypoints_{modality}"] = cam_kp_3d.astype(np.float32)
        # if modality == "lidar" and "gt_keypoints_pc_centered_input_lidar" in sample:
        #     center = cam_kp_3d.mean(axis=0, keepdims=True)
        #     centered = cam_kp_3d - center
        #     sample["gt_keypoints_pc_centered_input_lidar"] = centered.astype(np.float32)
        #     if "gt_keypoints_pc_center_lidar" in sample:
        #         sample["gt_keypoints_pc_center_lidar"] = center.squeeze(0).astype(np.float32)

    def _get_image_size(self, sample, modality):
        size_key = f"image_size_hw_{modality}"
        if size_key in sample:
            size = sample[size_key]
            if isinstance(size, (list, tuple)) and len(size) == 2 and not isinstance(size[0], (list, tuple)):
                return int(size[0]), int(size[1])
        if "image_size_hw" in sample:
            size = sample["image_size_hw"]
            if isinstance(size, (list, tuple)) and len(size) == 2 and not isinstance(size[0], (list, tuple)):
                return int(size[0]), int(size[1])
        input_key = f"input_{modality}"
        if input_key in sample:
            frames = sample[input_key]
            if isinstance(frames, (list, tuple)) and len(frames) > 0:
                h, w = frames[0].shape[:2]
                return int(h), int(w)
            if isinstance(frames, np.ndarray) and frames.ndim >= 3:
                return int(frames.shape[-3]), int(frames.shape[-2])
        return self.default_image_size_hw

    @staticmethod
    def _transform_to_camera(points, extrinsic):
        R = extrinsic[:, :3]
        T = extrinsic[:, 3]
        return (R @ points.T).T + T.reshape(1, 3)

    @staticmethod
    def _project_to_image(points, extrinsic, intrinsic):
        cam_points = ReplaceBySyntheticCamera._transform_to_camera(points, extrinsic)
        cam_z = np.clip(cam_points[:, 2], 1e-6, None)
        proj = (intrinsic @ cam_points.T).T
        u = proj[:, 0] / cam_z
        v = proj[:, 1] / cam_z
        return np.stack([u, v], axis=-1)

    @staticmethod
    def _normalize_2d(points_2d, image_size_hw):
        height, width = image_size_hw
        x = points_2d[..., 0] / (width - 1) * 2.0 - 1.0
        y = points_2d[..., 1] / (height - 1) * 2.0 - 1.0
        return np.stack([x, y], axis=-1)


class SyncKeypointsWithCameraEncoding:
    """Recompute per-modality keypoints from gt_camera encoding for consistency."""

    def __init__(
        self,
        modalities: tuple = None,
        normalize_2d: bool = True,
        default_image_size_hw: tuple = (224, 224),
        pose_encoding_type: str = "absT_quaR_FoV",
    ):
        self.modalities = modalities
        self.normalize_2d = bool(normalize_2d)
        self.default_image_size_hw = (int(default_image_size_hw[0]), int(default_image_size_hw[1]))
        self.pose_encoding_type = pose_encoding_type

    def __call__(self, sample):
        gt_keypoints = sample.get("gt_keypoints", None)
        if gt_keypoints is None:
            return sample

        if not isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = torch.as_tensor(gt_keypoints, dtype=torch.float32)

        modalities = self.modalities or sample.get("modalities", [])
        for modality in modalities:
            if modality.lower() not in {"rgb", "depth"}:
                continue
            cam_key = f"gt_camera_{modality}"
            if cam_key not in sample:
                continue
            gt_camera = sample[cam_key]
            if not isinstance(gt_camera, torch.Tensor):
                gt_camera = torch.as_tensor(gt_camera, dtype=torch.float32)

            image_size_hw = self._get_image_size(sample, modality)
            extrinsics, intrinsics = self._decode_camera(gt_camera, image_size_hw)
            if extrinsics is None or intrinsics is None:
                continue

            proj = self._project_to_image(gt_keypoints, extrinsics, intrinsics)
            if self.normalize_2d:
                proj = self._normalize_2d(proj, image_size_hw)
                proj = torch.clamp(proj, -1.0, 1.0)

            # Remove batch dim for per-sample storage
            if proj.dim() == 4 and proj.shape[0] == 1:
                proj = proj.squeeze(0)
            sample[f"gt_keypoints_2d_{modality}"] = proj

        return sample

    def _decode_camera(self, gt_camera, image_size_hw):
        if gt_camera.dim() == 2:
            gt_camera = gt_camera.unsqueeze(0)
        if gt_camera.dim() != 3:
            return None, None
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            gt_camera,
            image_size_hw=image_size_hw,
            pose_encoding_type=self.pose_encoding_type,
            build_intrinsics=True,
        )
        return extrinsics, intrinsics

    def _get_image_size(self, sample, modality):
        size_key = f"image_size_hw_{modality}"
        if size_key in sample:
            size = sample[size_key]
            if isinstance(size, (list, tuple)) and len(size) == 2 and not isinstance(size[0], (list, tuple)):
                return int(size[0]), int(size[1])
        if "image_size_hw" in sample:
            size = sample["image_size_hw"]
            if isinstance(size, (list, tuple)) and len(size) == 2 and not isinstance(size[0], (list, tuple)):
                return int(size[0]), int(size[1])
        input_key = f"input_{modality}"
        if input_key in sample:
            frames = sample[input_key]
            if isinstance(frames, (list, tuple)) and len(frames) > 0:
                h, w = frames[0].shape[:2]
                return int(h), int(w)
            if isinstance(frames, torch.Tensor) and frames.dim() >= 4:
                return int(frames.shape[-2]), int(frames.shape[-1])
            if isinstance(frames, np.ndarray) and frames.ndim >= 3:
                return int(frames.shape[-3]), int(frames.shape[-2])
        return self.default_image_size_hw

    @staticmethod
    def _project_to_image(points, extrinsics, intrinsics):
        # points: (J,3) or (S,J,3), extrinsics/intrinsics: (B,S,3,4)/(B,S,3,3)
        if points.dim() == 2:
            points = points.unsqueeze(0).unsqueeze(0)
        elif points.dim() == 3:
            points = points.unsqueeze(0)
        B, S, J, _ = points.shape
        extrinsics = extrinsics[:B, :S]
        intrinsics = intrinsics[:B, :S]
        R = extrinsics[..., :3, :3]
        T = extrinsics[..., :3, 3]
        cam_points = torch.einsum("bsij,bsnj->bsni", R, points) + T.unsqueeze(2)
        cam_z = cam_points[..., 2].clamp(min=1e-6)
        proj = torch.einsum("bsij,bsnj->bsni", intrinsics, cam_points)
        u = proj[..., 0] / cam_z
        v = proj[..., 1] / cam_z
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _normalize_2d(points_2d, image_size_hw):
        height, width = image_size_hw
        x = points_2d[..., 0] / (width - 1) * 2.0 - 1.0
        y = points_2d[..., 1] / (height - 1) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)
