import numpy as np
import torch

from misc.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


def _camera_to_list(camera):
    if camera is None:
        return []
    if isinstance(camera, (list, tuple)):
        return list(camera)
    return [camera]


def _restore_camera_type(cameras, original):
    if isinstance(original, (list, tuple)):
        return cameras
    return cameras[0] if cameras else None


def _split_frames_by_view(frames):
    if frames is None:
        return []
    if isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            return []
        if isinstance(frames[0], (list, tuple)):
            return [list(view_frames) for view_frames in frames]
        return [list(frames)]
    if isinstance(frames, torch.Tensor):
        if frames.dim() >= 5:
            return [frames[i] for i in range(frames.shape[0])]
        return [frames]
    if isinstance(frames, np.ndarray):
        if frames.ndim >= 5:
            return [frames[i] for i in range(frames.shape[0])]
        return [frames]
    return []


def _sequence_len(frames):
    if isinstance(frames, (list, tuple)):
        return len(frames)
    if isinstance(frames, torch.Tensor):
        return int(frames.shape[0]) if frames.dim() >= 1 else 0
    if isinstance(frames, np.ndarray):
        return int(frames.shape[0]) if frames.ndim >= 1 else 0
    return 0


def _frame_hw_from_view(frames):
    if isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            return None
        frame0 = frames[0]
        if hasattr(frame0, "shape") and len(frame0.shape) >= 2:
            shape = frame0.shape
            if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[1] > 4:
                return int(shape[1]), int(shape[2])
            return int(shape[0]), int(shape[1])
        return None
    if isinstance(frames, torch.Tensor):
        if frames.dim() >= 5:
            return int(frames.shape[-2]), int(frames.shape[-1])
        if frames.dim() == 4:
            if frames.shape[-1] in (1, 3, 4):
                return int(frames.shape[-3]), int(frames.shape[-2])
            return int(frames.shape[-2]), int(frames.shape[-1])
        if frames.dim() == 3:
            if frames.shape[0] in (1, 3, 4):
                return int(frames.shape[1]), int(frames.shape[2])
            return int(frames.shape[-2]), int(frames.shape[-1])
        return None
    if isinstance(frames, np.ndarray):
        if frames.ndim >= 5:
            return int(frames.shape[-3]), int(frames.shape[-2])
        if frames.ndim == 4:
            if frames.shape[-1] in (1, 3, 4):
                return int(frames.shape[-3]), int(frames.shape[-2])
            return int(frames.shape[-2]), int(frames.shape[-1])
        if frames.ndim == 3:
            if frames.shape[-1] in (1, 3, 4):
                return int(frames.shape[0]), int(frames.shape[1])
            return int(frames.shape[-2]), int(frames.shape[-1])
        return None
    return None


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

            camera_list = _camera_to_list(camera)
            frame_views = _split_frames_by_view(frames)
            if not camera_list or not frame_views:
                continue

            if len(frame_views) == 1 and len(camera_list) > 1:
                frame_views = frame_views * len(camera_list)
            elif len(frame_views) != len(camera_list):
                n = min(len(frame_views), len(camera_list))
                camera_list = camera_list[:n]
                frame_views = frame_views[:n]

            pose_enc_views = []
            for camera_view, frames_view in zip(camera_list, frame_views):
                seq_len = _sequence_len(frames_view)
                if seq_len <= 0:
                    continue
                hw = _frame_hw_from_view(frames_view)
                if hw is None:
                    continue
                extrinsic = np.asarray(camera_view["extrinsic"], dtype=np.float32)
                intrinsic = np.asarray(camera_view["intrinsic"], dtype=np.float32)

                extrinsics = torch.from_numpy(np.stack([extrinsic] * seq_len, axis=0)).unsqueeze(0)
                intrinsics = torch.from_numpy(np.stack([intrinsic] * seq_len, axis=0)).unsqueeze(0)
                pose_enc = extri_intri_to_pose_encoding(
                    extrinsics,
                    intrinsics,
                    image_size_hw=hw,
                    pose_encoding_type=self.pose_encoding_type,
                )
                pose_enc_views.append(pose_enc.squeeze(0))

            if not pose_enc_views:
                continue
            if len(pose_enc_views) == 1:
                sample[f"gt_camera_{modality}"] = pose_enc_views[0]
            else:
                sample[f"gt_camera_{modality}"] = torch.stack(pose_enc_views, dim=0)

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
            camera_raw = sample.get(camera_key)
            camera_list = _camera_to_list(camera_raw)
            if not camera_list:
                continue

            if gt_keypoints is not None:
                self._update_keypoints(sample, modality, gt_keypoints, camera_list)

            if not apply_synth:
                continue

            synthetic_cameras = []
            for camera in camera_list:
                intrinsic = np.asarray(camera["intrinsic"], dtype=np.float32)
                extrinsic = np.asarray(camera["extrinsic"], dtype=np.float32)
                synthetic_cameras.append(self._jitter_camera(intrinsic, extrinsic))
            sample[camera_key] = _restore_camera_type(synthetic_cameras, camera_raw)

            if gt_keypoints is not None:
                self._update_keypoints(sample, modality, gt_keypoints, synthetic_cameras)

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

    def _update_keypoints(self, sample, modality, gt_keypoints, cameras):
        if modality in {"rgb", "depth"}:
            image_size_hw = self._get_image_size(sample, modality)
            points = np.asarray(gt_keypoints, dtype=np.float32)
            kp_2d_views = []
            for camera in cameras:
                extrinsic = np.asarray(camera["extrinsic"], dtype=np.float32)
                intrinsic = np.asarray(camera["intrinsic"], dtype=np.float32)
                kp_2d = self._project_to_image(points, extrinsic, intrinsic)
                if self.normalize_2d:
                    kp_2d = self._normalize_2d(kp_2d, image_size_hw)
                    kp_2d = np.clip(kp_2d, -1.0, 1.0)
                kp_2d_views.append(kp_2d.astype(np.float32))
            if len(kp_2d_views) == 1:
                sample[f"gt_keypoints_2d_{modality}"] = kp_2d_views[0]
            elif kp_2d_views:
                sample[f"gt_keypoints_2d_{modality}"] = np.stack(kp_2d_views, axis=0)
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
                frame0 = frames[0]
                if isinstance(frame0, (list, tuple)) and len(frame0) > 0:
                    frame0 = frame0[0]
                if hasattr(frame0, "shape") and len(frame0.shape) >= 2:
                    shape = frame0.shape
                    if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[1] > 4:
                        return int(shape[1]), int(shape[2])
                    return int(shape[0]), int(shape[1])
            if isinstance(frames, torch.Tensor) and frames.dim() >= 4:
                return int(frames.shape[-2]), int(frames.shape[-1])
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
        if gt_camera.dim() == 1:
            gt_camera = gt_camera.unsqueeze(0).unsqueeze(0)
        if gt_camera.dim() == 2:
            gt_camera = gt_camera.unsqueeze(0)
        if gt_camera.dim() == 3:
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                gt_camera,
                image_size_hw=image_size_hw,
                pose_encoding_type=self.pose_encoding_type,
                build_intrinsics=True,
            )
            return extrinsics, intrinsics
        if gt_camera.dim() == 4:
            b, v, s, d = gt_camera.shape
            gt_camera = gt_camera.reshape(b * v, s, d)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                gt_camera,
                image_size_hw=image_size_hw,
                pose_encoding_type=self.pose_encoding_type,
                build_intrinsics=True,
            )
            extrinsics = extrinsics.reshape(b, v, s, 3, 4)
            intrinsics = intrinsics.reshape(b, v, s, 3, 3)
            return extrinsics, intrinsics
        if gt_camera.dim() != 3:
            return None, None
        return None, None

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
                frame0 = frames[0]
                if isinstance(frame0, (list, tuple)) and len(frame0) > 0:
                    frame0 = frame0[0]
                if hasattr(frame0, "shape") and len(frame0.shape) >= 2:
                    shape = frame0.shape
                    if len(shape) == 3 and shape[0] in (1, 3, 4) and shape[1] > 4:
                        return int(shape[1]), int(shape[2])
                    return int(shape[0]), int(shape[1])
            if isinstance(frames, torch.Tensor) and frames.dim() >= 4:
                return int(frames.shape[-2]), int(frames.shape[-1])
            if isinstance(frames, np.ndarray) and frames.ndim >= 3:
                return int(frames.shape[-3]), int(frames.shape[-2])
        return self.default_image_size_hw

    @staticmethod
    def _project_to_image(points, extrinsics, intrinsics):
        # points: (J,3), (S,J,3), (L,J,3), or (L,S,J,3)
        # extrinsics/intrinsics: (...,S,3,4)/(...,S,3,3)
        leading_shape = extrinsics.shape[:-3]
        seq_len = extrinsics.shape[-3]
        leading_size = int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1

        extrinsics_flat = extrinsics.reshape(leading_size, seq_len, 3, 4)
        intrinsics_flat = intrinsics.reshape(leading_size, seq_len, 3, 3)

        if points.dim() == 2:
            points = points.unsqueeze(0).unsqueeze(0)
        elif points.dim() == 3:
            if points.shape[0] == seq_len:
                points = points.unsqueeze(0)
            elif points.shape[0] in {1, leading_size}:
                points = points.unsqueeze(1)
            else:
                raise ValueError(
                    f"Ambiguous keypoint shape {tuple(points.shape)} for camera shape "
                    f"(L={leading_size}, S={seq_len})."
                )
        elif points.dim() != 4:
            raise ValueError(f"Expected points dim in [2, 3, 4], got {points.dim()}.")

        if points.shape[0] == 1 and leading_size > 1:
            points = points.expand(leading_size, -1, -1, -1)
        if points.shape[1] == 1 and seq_len > 1:
            points = points.expand(-1, seq_len, -1, -1)

        if points.shape[0] != leading_size or points.shape[1] != seq_len:
            raise ValueError(
                f"Keypoints shape {tuple(points.shape)} does not match camera shape "
                f"(L={leading_size}, S={seq_len})."
            )

        R = extrinsics_flat[..., :3, :3]
        T = extrinsics_flat[..., :3, 3]
        cam_points = torch.einsum("lsij,lsnj->lsni", R, points) + T.unsqueeze(2)
        cam_z = cam_points[..., 2].clamp(min=1e-6)
        proj = torch.einsum("lsij,lsnj->lsni", intrinsics_flat, cam_points)
        u = proj[..., 0] / cam_z
        v = proj[..., 1] / cam_z
        out = torch.stack([u, v], dim=-1)
        return out.reshape(*leading_shape, seq_len, out.shape[-2], 2)

    @staticmethod
    def _normalize_2d(points_2d, image_size_hw):
        height, width = image_size_hw
        x = points_2d[..., 0] / (width - 1) * 2.0 - 1.0
        y = points_2d[..., 1] / (height - 1) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)
