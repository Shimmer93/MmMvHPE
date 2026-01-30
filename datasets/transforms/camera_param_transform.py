import numpy as np
import torch

from misc.pose_enc import extri_intri_to_pose_encoding


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
