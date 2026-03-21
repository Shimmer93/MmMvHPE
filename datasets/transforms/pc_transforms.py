import numpy as np
import torch
from copy import deepcopy
from typing import Optional, Sequence, Union, List
from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
# from miniball import get_bounding_ball

def initialize_affine_matrix():
    return np.eye(4, dtype=np.float32)

class PCPad():
    def __init__(self, 
                 num_points: int, 
                 pad_mode: str = 'zero',
                 keys: List[str] = ['input_mmwave']):
        assert pad_mode in ['zero', 'repeat'], "pad_mode must be 'zero' or 'repeat'"
        self.num_points = num_points
        self.pad_mode = pad_mode
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            pc_seq = results[key]
            is_multi_view = (
                isinstance(pc_seq, (list, tuple))
                and len(pc_seq) > 0
                and isinstance(pc_seq[0], (list, tuple))
            )
            if is_multi_view:
                padded_views = []
                for view_seq in pc_seq:
                    padded_view = []
                    for pc in view_seq:
                        padded_view.append(self._pad_single_pc(pc))
                    padded_views.append(padded_view)
                results[key] = padded_views
            else:
                padded_pc_seq = []
                for pc in pc_seq:
                    padded_pc_seq.append(self._pad_single_pc(pc))
                results[key] = padded_pc_seq

        return results

    def _pad_single_pc(self, pc):
        N, C = pc.shape

        if N == 0:
            return np.random.normal(0, 1, (self.num_points, C)).astype(pc.dtype)

        if N < self.num_points:
            if self.pad_mode == 'zero':
                pad = np.zeros((self.num_points - N, C), dtype=pc.dtype)
            elif self.pad_mode == 'repeat':
                np.random.shuffle(pc)
                repeat_times = (self.num_points - N) // N + 1
                pad = np.tile(pc, (repeat_times, 1))[:self.num_points - N]
            return np.vstack((pc, pad))

        if N > self.num_points:
            np.random.shuffle(pc)
            return pc[:self.num_points]

        return pc
    
class PCJitter():
    def __init__(self, 
                 sigma: float = 0.01, 
                 clip: Optional[float] = None,
                 keys: List[str] = ['input_mmwave']):
        self.sigma = sigma
        self.clip = clip
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            pc_seq = results[key]
            jittered_pc_seq = []

            for pc in pc_seq:
                N, C = pc.shape
                jitter = np.random.normal(0, self.sigma, (N, 3))
                if self.clip is not None:
                    jitter = np.clip(jitter, -self.clip, self.clip)
                pc[:, :3] += jitter
                jittered_pc_seq.append(pc)

            results[key] = jittered_pc_seq

        return results
    
class PCTranslate():
    def __init__(self, 
                 translate_range: Union[float, Sequence[float]] = 0.2,
                 keys: List[str] = ['input_mmwave']):
        if isinstance(translate_range, float):
            self.translate_range = (-translate_range, translate_range)
        else:
            assert len(translate_range) == 2, "translate_range must be a float or a tuple of two floats"
            self.translate_range = translate_range
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            pc_seq = results[key]
            translated_pc_seq = []

            translation = np.random.uniform(self.translate_range[0], self.translate_range[1], size=(1, 3))
            
            for pc in pc_seq:
                pc[:, :3] += translation
                translated_pc_seq.append(pc)

            results[key] = translated_pc_seq
            if f'{key}_affine' not in results:
                results[f'{key}_affine'] = initialize_affine_matrix()
            results[f'{key}_affine'][:3, 3] += translation.flatten()

        return results
    
class PCScale():
    def __init__(self, 
                 scale_range: Union[float, Sequence[float]] = 0.2,
                 keys: List[str] = ['input_mmwave']):
        if isinstance(scale_range, float):
            self.scale_range = (1 - scale_range, 1 + scale_range)
        else:
            assert len(scale_range) == 2, "scale_range must be a float or a tuple of two floats"
            self.scale_range = scale_range
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            pc_seq = results[key]
            scaled_pc_seq = []

            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            
            for pc in pc_seq:
                pc[:, :3] *= scale
                scaled_pc_seq.append(pc)

            results[key] = scaled_pc_seq
            if f'{key}_affine' not in results:
                results[f'{key}_affine'] = initialize_affine_matrix()
            results[f'{key}_affine'][:3, :3] = scale * results[f'{key}_affine'][:3, :3]

        return results

class PCRotate():
    def __init__(self, 
                 rotate_range: Union[float, Sequence[float]] = 30.0,
                 keys: List[str] = ['input_mmwave']):
        if isinstance(rotate_range, float):
            self.rotate_range = (-rotate_range, rotate_range)
        else:
            assert len(rotate_range) == 2, "rotate_range must be a float or a tuple of two floats"
            self.rotate_range = rotate_range
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            pc_seq = results[key]
            rotated_pc_seq = []

            angle = np.random.uniform(self.rotate_range[0], self.rotate_range[1]) * np.pi / 180.0
            cosval = np.cos(angle)
            sinval = np.sin(angle)
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0, 1]])

            for pc in pc_seq:
                pc[:, :3] = pc[:, :3].dot(rotation_matrix.T)
                rotated_pc_seq.append(pc)

            results[key] = rotated_pc_seq
            if f'{key}_affine' not in results:
                results[f'{key}_affine'] = initialize_affine_matrix()
            results[f'{key}_affine'][:3, :3] = rotation_matrix.dot(results[f'{key}_affine'][:3, :3])

        return results
    
class PCDropout():
    def __init__(self, 
                 max_dropout_ratio: float = 0.2,
                 keys: List[str] = ['input_mmwave']):
        assert 0.0 <= max_dropout_ratio < 1.0, "max_dropout_ratio must be in [0.0, 1.0)"
        self.max_dropout_ratio = max_dropout_ratio
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            pc_seq = results[key]
            dropped_pc_seq = []

            for pc in pc_seq:
                dropout_ratio = np.random.uniform(0, self.max_dropout_ratio)
                N, C = pc.shape
                drop_idx = np.random.choice(N, int(N * dropout_ratio), replace=False)
                pc[drop_idx, :3] = pc[0, :3]  # set to the first point
                dropped_pc_seq.append(pc)

            results[key] = dropped_pc_seq

        return results


class PCStructuredOcclusionAug():
    def __init__(
        self,
        apply_prob: float = 0.35,
        range_image_size: Union[int, Sequence[int]] = (64, 256),
        blob_count_range: Union[int, Sequence[int]] = (1, 2),
        blob_shape_mode: str = 'mixed',
        rectangle_height_ratio_range: Sequence[float] = (0.08, 0.18),
        rectangle_width_ratio_range: Sequence[float] = (0.08, 0.18),
        circle_radius_ratio_range: Sequence[float] = (0.06, 0.12),
        min_points_kept: int = 256,
        keys: List[str] = ['input_lidar'],
    ):
        if not 0.0 <= float(apply_prob) <= 1.0:
            raise ValueError("apply_prob must be in [0.0, 1.0].")
        self.apply_prob = float(apply_prob)

        if isinstance(range_image_size, int):
            self.range_image_size = (int(range_image_size), int(range_image_size))
        else:
            if len(range_image_size) != 2:
                raise ValueError("range_image_size must be an int or a 2-tuple/list.")
            self.range_image_size = (int(range_image_size[0]), int(range_image_size[1]))
        if self.range_image_size[0] <= 1 or self.range_image_size[1] <= 1:
            raise ValueError("range_image_size dimensions must both be > 1.")

        self.blob_count_range = self._normalize_int_range(blob_count_range, "blob_count_range", min_value=1)
        self.blob_shape_mode = str(blob_shape_mode)
        if self.blob_shape_mode not in ['rectangle', 'circle', 'mixed']:
            raise ValueError("blob_shape_mode must be one of ['rectangle', 'circle', 'mixed'].")

        self.rectangle_height_ratio_range = self._normalize_float_range(
            rectangle_height_ratio_range, "rectangle_height_ratio_range"
        )
        self.rectangle_width_ratio_range = self._normalize_float_range(
            rectangle_width_ratio_range, "rectangle_width_ratio_range"
        )
        self.circle_radius_ratio_range = self._normalize_float_range(
            circle_radius_ratio_range, "circle_radius_ratio_range"
        )
        self.min_points_kept = int(min_points_kept)
        if self.min_points_kept < 1:
            raise ValueError("min_points_kept must be >= 1.")
        self.keys = keys

    @staticmethod
    def _normalize_int_range(value, name: str, min_value: Optional[int] = None):
        if isinstance(value, int):
            low = high = int(value)
        else:
            if len(value) != 2:
                raise ValueError(f"{name} must be an int or a 2-tuple/list.")
            low, high = int(value[0]), int(value[1])
        if low > high:
            raise ValueError(f"{name} low must be <= high.")
        if min_value is not None and low < min_value:
            raise ValueError(f"{name} values must be >= {min_value}.")
        return (low, high)

    @staticmethod
    def _normalize_float_range(value, name: str):
        if len(value) != 2:
            raise ValueError(f"{name} must be a 2-tuple/list.")
        low, high = float(value[0]), float(value[1])
        if low > high:
            raise ValueError(f"{name} low must be <= high.")
        if low <= 0.0 or high > 1.0:
            raise ValueError(f"{name} values must be in (0.0, 1.0].")
        return (low, high)

    def _sample_blob_count(self) -> int:
        low, high = self.blob_count_range
        return int(np.random.randint(low, high + 1))

    def _sample_shape(self) -> str:
        if self.blob_shape_mode != 'mixed':
            return self.blob_shape_mode
        return 'rectangle' if np.random.rand() < 0.5 else 'circle'

    def _compute_range_image_coords(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        horizontal_range = np.sqrt(np.maximum(x * x + z * z, 1e-12))
        azimuth = np.arctan2(z, x)
        elevation = np.arctan2(y, horizontal_range)

        az_min, az_max = float(np.min(azimuth)), float(np.max(azimuth))
        el_min, el_max = float(np.min(elevation)), float(np.max(elevation))
        if (az_max - az_min) < 1e-6 or (el_max - el_min) < 1e-6:
            raise ValueError("Point cloud angular support is degenerate; cannot build range-image coordinates.")

        h, w = self.range_image_size
        cols = (azimuth - az_min) / (az_max - az_min)
        rows = (elevation - el_min) / (el_max - el_min)
        cols = np.clip(np.floor(cols * (w - 1)), 0, w - 1).astype(np.int32)
        rows = np.clip(np.floor(rows * (h - 1)), 0, h - 1).astype(np.int32)
        return rows, cols

    def _sample_rectangle_mask(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        h, w = self.range_image_size
        center_row = int(np.random.randint(0, h))
        center_col = int(np.random.randint(0, w))
        half_h = max(1, height // 2)
        half_w = max(1, width // 2)
        row_min = max(0, center_row - half_h)
        row_max = min(h - 1, center_row + half_h)
        col_min = max(0, center_col - half_w)
        col_max = min(w - 1, center_col + half_w)
        return (
            (rows >= row_min)
            & (rows <= row_max)
            & (cols >= col_min)
            & (cols <= col_max)
        )

    def _sample_circle_mask(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        h, w = self.range_image_size
        center_row = float(np.random.randint(0, h))
        center_col = float(np.random.randint(0, w))
        dist_sq = (rows.astype(np.float32) - center_row) ** 2 + (cols.astype(np.float32) - center_col) ** 2
        return dist_sq <= float(radius) ** 2

    def _augment_single_pc(self, pc: np.ndarray) -> np.ndarray:
        if not isinstance(pc, np.ndarray):
            raise ValueError(f"Point cloud must be np.ndarray, got {type(pc).__name__}.")
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must have shape (N, C>=3), got {pc.shape}.")
        if pc.shape[0] == 0 or np.random.rand() > self.apply_prob:
            return pc

        xyz = pc[:, :3]
        if not np.all(np.isfinite(xyz)):
            raise ValueError("Point cloud contains non-finite xyz coordinates.")

        try:
            rows, cols = self._compute_range_image_coords(xyz.astype(np.float32))
        except ValueError:
            return pc

        drop_mask = np.zeros(pc.shape[0], dtype=bool)
        h, w = self.range_image_size
        for _ in range(self._sample_blob_count()):
            shape = self._sample_shape()
            if shape == 'rectangle':
                rect_h = max(1, int(round(np.random.uniform(*self.rectangle_height_ratio_range) * h)))
                rect_w = max(1, int(round(np.random.uniform(*self.rectangle_width_ratio_range) * w)))
                drop_mask |= self._sample_rectangle_mask(rows, cols, rect_h, rect_w)
            else:
                radius = max(1.0, np.random.uniform(*self.circle_radius_ratio_range) * min(h, w))
                drop_mask |= self._sample_circle_mask(rows, cols, radius)

        keep_mask = ~drop_mask
        kept_indices = np.flatnonzero(keep_mask)
        if kept_indices.size == 0:
            keep_count = min(self.min_points_kept, pc.shape[0])
            kept_indices = np.random.choice(pc.shape[0], size=keep_count, replace=False)
            keep_mask = np.zeros(pc.shape[0], dtype=bool)
            keep_mask[kept_indices] = True
        elif kept_indices.size < min(self.min_points_kept, pc.shape[0]):
            keep_target = min(self.min_points_kept, pc.shape[0])
            restore_candidates = np.flatnonzero(~keep_mask)
            restore_count = min(keep_target - kept_indices.size, restore_candidates.size)
            if restore_count > 0:
                restored = np.random.choice(restore_candidates, size=restore_count, replace=False)
                keep_mask[restored] = True

        return pc[keep_mask]

    def _apply_to_pc_seq(self, pc_seq):
        is_multi_view = (
            isinstance(pc_seq, (list, tuple))
            and len(pc_seq) > 0
            and isinstance(pc_seq[0], (list, tuple))
        )
        if is_multi_view:
            out = []
            for view_seq in pc_seq:
                out_view = []
                for pc in view_seq:
                    out_view.append(self._augment_single_pc(pc))
                out.append(out_view)
            return out

        out = []
        for pc in pc_seq:
            out.append(self._augment_single_pc(pc))
        return out

    def __call__(self, results):
        for key in self.keys:
            if key not in results:
                continue
            results[key] = self._apply_to_pc_seq(results[key])
        return results
    
class PCNormalize():
    def __init__(self, 
                 norm_type: str = 'mean_std',
                 keys: List[str] = ['input_mmwave']):
        assert norm_type in ['mean_std', 'min_max', 'mean', 'median'], \
                "norm_type must be one of 'mean_std', 'min_max', 'mean', 'median'"
        self.norm_type = norm_type
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            pc_seq = results[key]
            normalized_pc_seq = []

            all_ps = np.concatenate(pc_seq, axis=0)
            if self.norm_type == 'mean_std':
                center = np.mean(all_ps[:, :3], axis=0)
                radius = np.std(all_ps[:, :3], axis=0)
            elif self.norm_type == 'min_max':
                center = (np.min(all_ps[:, :3], axis=0) + np.max(all_ps[:, :3], axis=0)) / 2.0
                radius = (np.max(all_ps[:, :3], axis=0) - np.min(all_ps[:, :3], axis=0)) / 2.0
            elif self.norm_type == 'mean':
                center = np.mean(all_ps[:, :3], axis=0)
                radius = 1.0
            elif self.norm_type == 'median':
                center = np.median(all_ps[:, :3], axis=0)
                radius = 1.0
            else:
                raise ValueError(f"Unknown norm_type: {self.norm_type}")
            
            for pc in pc_seq:
                pc[:, :3] = (pc[:, :3] - center) / radius
                normalized_pc_seq.append(pc)

            results[key] = normalized_pc_seq
            if f'{key}_affine' not in results:
                results[f'{key}_affine'] = initialize_affine_matrix()
            results[f'{key}_affine'][:3, :3] /= radius
            results[f'{key}_affine'][:3, 3] = (results[f'{key}_affine'][:3, 3] - center) / radius

        return results

class PCCenterWithKeypoints():
    def __init__(
        self,
        center_type: str = 'mean',
        keys: List[str] = ['input_lidar'],
        keypoints_key: str = 'gt_keypoints',
        shifted_keypoints_suffix: str = '_pc_centered',
        root_joint_indices: List[int] = [0, 2],
        centered_root_tol: float = 1e-4,
    ):
        assert center_type in ['mean', 'median'], "center_type must be 'mean' or 'median'"
        self.center_type = center_type
        self.keys = keys
        self.keypoints_key = keypoints_key
        self.shifted_keypoints_suffix = shifted_keypoints_suffix
        if not isinstance(root_joint_indices, (list, tuple)) or len(root_joint_indices) == 0:
            raise ValueError("root_joint_indices must be a non-empty list/tuple of ints.")
        self.root_joint_indices = [int(i) for i in root_joint_indices]
        self.centered_root_tol = float(centered_root_tol)

    @staticmethod
    def _modality_from_key(key: str) -> str:
        if key.startswith("input_"):
            return key[len("input_"):]
        return key

    @staticmethod
    def _apply_extrinsic(points, extrinsic):
        if points is None:
            return None
        if isinstance(points, torch.Tensor):
            R = torch.as_tensor(extrinsic[:, :3], dtype=points.dtype, device=points.device)
            T = torch.as_tensor(extrinsic[:, 3], dtype=points.dtype, device=points.device)
            shape = points.shape
            pts = points.reshape(-1, 3)
            pts = (pts @ R.t()) + T
            return pts.reshape(shape)
        pts = np.asarray(points, dtype=np.float32)
        shape = pts.shape
        pts = pts.reshape(-1, 3)
        R = extrinsic[:, :3].astype(np.float32)
        T = extrinsic[:, 3].astype(np.float32)
        pts = (R @ pts.T).T + T.reshape(1, 3)
        return pts.reshape(shape)

    @staticmethod
    def _to_camera_list(camera):
        if camera is None:
            return []
        if isinstance(camera, (list, tuple)):
            return list(camera)
        return [camera]

    @staticmethod
    def _restore_camera_type(cameras, original):
        if isinstance(original, (list, tuple)):
            return cameras
        return cameras[0] if cameras else None

    @staticmethod
    def _flatten_pc_sequence(pc_seq):
        if (
            isinstance(pc_seq, (list, tuple))
            and len(pc_seq) > 0
            and isinstance(pc_seq[0], (list, tuple))
        ):
            return [pc for view_seq in pc_seq for pc in view_seq]
        return list(pc_seq)

    @staticmethod
    def _center_pc_sequence(pc_seq, center):
        if (
            isinstance(pc_seq, (list, tuple))
            and len(pc_seq) > 0
            and isinstance(pc_seq[0], (list, tuple))
        ):
            out = []
            for view_seq in pc_seq:
                out_view = []
                for pc in view_seq:
                    pc[:, :3] = pc[:, :3] - center
                    out_view.append(pc)
                out.append(out_view)
            return out
        out = []
        for pc in pc_seq:
            pc[:, :3] = pc[:, :3] - center
            out.append(pc)
        return out

    @staticmethod
    def _is_pelvis_centered(keypoints, root_joint_indices: List[int], tol: float):
        if isinstance(keypoints, np.ndarray):
            if keypoints.ndim < 2 or keypoints.shape[-1] != 3:
                return False
            num_joints = keypoints.shape[-2]
            if num_joints <= 0:
                return False
            leading = (0,) * (keypoints.ndim - 2)
            for root_idx in root_joint_indices:
                if 0 <= root_idx < num_joints:
                    root = keypoints[leading + (root_idx, slice(None))]
                    if np.linalg.norm(root) < tol:
                        return True
            return False
        if isinstance(keypoints, torch.Tensor):
            if keypoints.dim() < 2 or keypoints.shape[-1] != 3:
                return False
            num_joints = int(keypoints.shape[-2])
            if num_joints <= 0:
                return False
            leading = (0,) * (keypoints.dim() - 2)
            for root_idx in root_joint_indices:
                if 0 <= root_idx < num_joints:
                    root = keypoints[leading + (root_idx, slice(None))]
                    if torch.linalg.norm(root).item() < tol:
                        return True
            return False
        return False

    @staticmethod
    def _subtract_center(keypoints, center):
        if isinstance(keypoints, torch.Tensor):
            out = keypoints.clone()
            out[..., :3] = out[..., :3] - torch.as_tensor(center, dtype=out.dtype, device=out.device)
            return out
        out = np.asarray(keypoints, dtype=np.float32).copy()
        out[..., :3] = out[..., :3] - center
        return out

    def _update_gt_camera_translation(self, gt_camera, translations):
        if gt_camera is None:
            return None
        trans = np.asarray(translations, dtype=np.float32)
        if trans.ndim == 1:
            trans = trans[None, :]
        num_views = trans.shape[0]

        if isinstance(gt_camera, torch.Tensor):
            gt_cam = gt_camera.clone()
            trans_t = torch.as_tensor(trans, dtype=gt_cam.dtype, device=gt_cam.device)
            if gt_cam.dim() == 1:
                gt_cam[:3] = trans_t[0]
            elif gt_cam.dim() == 2:
                if num_views > 1 and gt_cam.shape[0] == num_views and gt_cam.shape[-1] == 9:
                    gt_cam[:, :3] = trans_t
                else:
                    gt_cam[..., :3] = trans_t[0]
            elif gt_cam.dim() == 3:
                if num_views > 1 and gt_cam.shape[0] == num_views and gt_cam.shape[-1] == 9:
                    gt_cam[..., :3] = trans_t[:, None, :]
                elif num_views > 1 and gt_cam.shape[1] == num_views and gt_cam.shape[-1] == 9:
                    gt_cam[..., :3] = trans_t[None, :, :]
                else:
                    gt_cam[..., :3] = trans_t[0]
            elif gt_cam.dim() == 4:
                if num_views > 1 and gt_cam.shape[1] == num_views and gt_cam.shape[-1] == 9:
                    gt_cam[..., :3] = trans_t[None, :, None, :]
                elif num_views > 1 and gt_cam.shape[2] == num_views and gt_cam.shape[-1] == 9:
                    gt_cam[..., :3] = trans_t[None, None, :, :]
                else:
                    gt_cam[..., :3] = trans_t[0]
            return gt_cam

        gt_cam = np.asarray(gt_camera, dtype=np.float32).copy()
        if gt_cam.ndim == 1:
            gt_cam[:3] = trans[0]
        elif gt_cam.ndim == 2:
            if num_views > 1 and gt_cam.shape[0] == num_views and gt_cam.shape[-1] == 9:
                gt_cam[:, :3] = trans
            else:
                gt_cam[..., :3] = trans[0]
        elif gt_cam.ndim == 3:
            if num_views > 1 and gt_cam.shape[0] == num_views and gt_cam.shape[-1] == 9:
                gt_cam[..., :3] = trans[:, None, :]
            elif num_views > 1 and gt_cam.shape[1] == num_views and gt_cam.shape[-1] == 9:
                gt_cam[..., :3] = trans[None, :, :]
            else:
                gt_cam[..., :3] = trans[0]
        elif gt_cam.ndim == 4:
            if num_views > 1 and gt_cam.shape[1] == num_views and gt_cam.shape[-1] == 9:
                gt_cam[..., :3] = trans[None, :, None, :]
            elif num_views > 1 and gt_cam.shape[2] == num_views and gt_cam.shape[-1] == 9:
                gt_cam[..., :3] = trans[None, None, :, :]
            else:
                gt_cam[..., :3] = trans[0]
        return gt_cam

    def __call__(self, results):
        for key in self.keys:
            if key not in results:
                results[f'{key}_affine'] = initialize_affine_matrix()
                continue
            pc_seq = results[key]
            flat_pc = self._flatten_pc_sequence(pc_seq)
            valid_pc = [pc for pc in flat_pc if pc is not None and pc.size > 0]
            if not valid_pc:
                continue
            all_ps = np.concatenate(valid_pc, axis=0)
            if all_ps.size == 0:
                continue

            if self.center_type == 'mean':
                center = np.mean(all_ps[:, :3], axis=0)
            else:
                center = np.median(all_ps[:, :3], axis=0)
            center = center.astype(np.float32)

            modality = self._modality_from_key(key)
            camera_key = f"{modality}_camera"
            gt_camera_key = f"gt_camera_{modality}"
            camera_raw = results.get(camera_key)
            camera_list = self._to_camera_list(camera_raw)
            extrinsic_before_list = []
            for camera in camera_list:
                if camera is not None and "extrinsic" in camera:
                    extrinsic_before_list.append(np.asarray(camera["extrinsic"], dtype=np.float32))

            results[key] = self._center_pc_sequence(pc_seq, center)
            if f'{key}_affine' not in results:
                results[f'{key}_affine'] = initialize_affine_matrix()
            results[f'{key}_affine'][:3, 3] = results[f'{key}_affine'][:3, 3] - center

            if self.keypoints_key in results and results[self.keypoints_key] is not None:
                keypoints = results[self.keypoints_key]
                if extrinsic_before_list:
                    shifted_views = []
                    for extrinsic_before in extrinsic_before_list:
                        keypoints_for_pc = keypoints
                        if self._is_pelvis_centered(
                            keypoints,
                            root_joint_indices=self.root_joint_indices,
                            tol=self.centered_root_tol,
                        ):
                            keypoints_for_pc = self._apply_extrinsic(keypoints_for_pc, extrinsic_before)
                        shifted_views.append(self._subtract_center(keypoints_for_pc, center))
                    if len(shifted_views) == 1:
                        shifted = shifted_views[0]
                    elif isinstance(shifted_views[0], torch.Tensor):
                        shifted = torch.stack(shifted_views, dim=0)
                    else:
                        shifted = np.stack(shifted_views, axis=0)
                else:
                    shifted = self._subtract_center(keypoints, center)
                results[f'{self.keypoints_key}{self.shifted_keypoints_suffix}_{key}'] = shifted

            if extrinsic_before_list:
                updated_camera_list = []
                translations = []
                for camera, extrinsic_before in zip(camera_list, extrinsic_before_list):
                    extrinsic_after = extrinsic_before.copy()
                    extrinsic_after[:, 3] = extrinsic_after[:, 3] - center
                    camera_out = dict(camera)
                    camera_out["extrinsic"] = extrinsic_after.astype(np.float32)
                    updated_camera_list.append(camera_out)
                    translations.append(extrinsic_after[:, 3])
                results[camera_key] = self._restore_camera_type(updated_camera_list, camera_raw)

                if gt_camera_key in results and results[gt_camera_key] is not None:
                    results[gt_camera_key] = self._update_gt_camera_translation(
                        results[gt_camera_key], np.stack(translations, axis=0)
                    )

        return results


class PCPCAAlign():
    """Align point clouds to principal axes and track the rotation in affine."""

    def __init__(
        self,
        keys: List[str] = ['input_lidar'],
        rotate_centered_keypoints: bool = True,
        centered_keypoints_prefix: str = 'gt_keypoints_pc_centered_',
        min_points: int = 32,
    ):
        self.keys = keys
        self.rotate_centered_keypoints = bool(rotate_centered_keypoints)
        self.centered_keypoints_prefix = centered_keypoints_prefix
        self.min_points = int(min_points)

    @staticmethod
    def _flatten_pc_sequence(pc_seq, key_name: str):
        if not isinstance(pc_seq, (list, tuple)):
            raise ValueError(f"`{key_name}` must be list/tuple before ToTensor, got {type(pc_seq).__name__}.")
        if len(pc_seq) == 0:
            return []

        flat = []
        is_multi_view = isinstance(pc_seq[0], (list, tuple))
        if is_multi_view:
            for view_idx, view_seq in enumerate(pc_seq):
                if not isinstance(view_seq, (list, tuple)):
                    raise ValueError(f"`{key_name}` view {view_idx} must be list/tuple.")
                for frame_idx, pc in enumerate(view_seq):
                    if not isinstance(pc, np.ndarray):
                        raise ValueError(
                            f"`{key_name}` view {view_idx} frame {frame_idx} must be np.ndarray, "
                            f"got {type(pc).__name__}."
                        )
                    if pc.ndim != 2 or pc.shape[1] < 3:
                        raise ValueError(
                            f"`{key_name}` view {view_idx} frame {frame_idx} must be (N, C>=3), got {pc.shape}."
                        )
                    if pc.shape[0] > 0:
                        flat.append(pc[:, :3])
        else:
            for frame_idx, pc in enumerate(pc_seq):
                if not isinstance(pc, np.ndarray):
                    raise ValueError(f"`{key_name}` frame {frame_idx} must be np.ndarray, got {type(pc).__name__}.")
                if pc.ndim != 2 or pc.shape[1] < 3:
                    raise ValueError(f"`{key_name}` frame {frame_idx} must be (N, C>=3), got {pc.shape}.")
                if pc.shape[0] > 0:
                    flat.append(pc[:, :3])
        return flat

    @staticmethod
    def _rotate_pc_sequence(pc_seq, rotation):
        is_multi_view = (
            isinstance(pc_seq, (list, tuple))
            and len(pc_seq) > 0
            and isinstance(pc_seq[0], (list, tuple))
        )
        if is_multi_view:
            rotated = []
            for view_seq in pc_seq:
                out_view = []
                for pc in view_seq:
                    pc_out = pc.copy()
                    pc_out[:, :3] = pc_out[:, :3].dot(rotation.T)
                    out_view.append(pc_out)
                rotated.append(out_view)
            return rotated

        rotated = []
        for pc in pc_seq:
            pc_out = pc.copy()
            pc_out[:, :3] = pc_out[:, :3].dot(rotation.T)
            rotated.append(pc_out)
        return rotated

    @staticmethod
    def _resolve_rotation(points_xyz):
        if points_xyz.shape[0] < 3:
            return None
        centered = points_xyz - points_xyz.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        if not np.all(np.isfinite(cov)):
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        for i in range(3):
            axis = eigvecs[:, i]
            dominant = int(np.argmax(np.abs(axis)))
            if axis[dominant] < 0:
                eigvecs[:, i] = -axis

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1.0

        rotation = eigvecs.T.astype(np.float32)
        if not np.all(np.isfinite(rotation)):
            return None
        return rotation

    @staticmethod
    def _rotate_keypoints_data(keypoints, rotation, key_name):
        if isinstance(keypoints, np.ndarray):
            if keypoints.shape[-1] != 3:
                raise ValueError(f"`{key_name}` must end with xyz=3, got {keypoints.shape}.")
            out = keypoints.copy()
            xyz = out[..., :3].reshape(-1, 3)
            out[..., :3] = xyz.dot(rotation.T).reshape(out[..., :3].shape)
            return out

        if isinstance(keypoints, torch.Tensor):
            if keypoints.shape[-1] != 3:
                raise ValueError(f"`{key_name}` must end with xyz=3, got {tuple(keypoints.shape)}.")
            out = keypoints.clone()
            rot = torch.as_tensor(rotation, dtype=out.dtype, device=out.device)
            xyz = out[..., :3].reshape(-1, 3)
            out[..., :3] = (xyz @ rot.t()).reshape(out[..., :3].shape)
            return out

        raise ValueError(f"`{key_name}` must be np.ndarray or torch.Tensor, got {type(keypoints).__name__}.")

    def __call__(self, results):
        for key in self.keys:
            if key not in results:
                continue

            pc_seq = results[key]
            flat_xyz = self._flatten_pc_sequence(pc_seq, key)
            if not flat_xyz:
                continue
            points_xyz = np.concatenate(flat_xyz, axis=0).astype(np.float32)
            if points_xyz.shape[0] < self.min_points:
                continue

            rotation = self._resolve_rotation(points_xyz)
            if rotation is None:
                continue

            results[key] = self._rotate_pc_sequence(pc_seq, rotation)
            affine_key = f'{key}_affine'
            if affine_key not in results:
                results[affine_key] = initialize_affine_matrix()
            results[affine_key][:3, :3] = rotation.dot(results[affine_key][:3, :3])

            if self.rotate_centered_keypoints:
                kp_key = f'{self.centered_keypoints_prefix}{key}'
                if kp_key in results and results[kp_key] is not None:
                    results[kp_key] = self._rotate_keypoints_data(results[kp_key], rotation, kp_key)

        return results
    
class PCRemoveOutliers():
    def __init__(self, 
                 outlier_method: str = 'radius',
                 **kwargs):
        
        assert outlier_method in ['radius', 'dbscan', 'box'], \
            "outlier_method must be one of 'radius', 'dbscan', 'box'"
        self.outlier_method = outlier_method
        self.kwargs = kwargs

    def __call__(self, results):
        for key in results.keys():
            pc_seq = results[key]
            filtered_pc_seq = []

            for pc in pc_seq:
                if self.outlier_method == 'radius':
                    center = np.mean(pc[:, :3], axis=0)
                    dists = np.linalg.norm(pc[:, :3] - center, axis=1)
                    radius_threshold = self.kwargs.get('radius_threshold', 3.0) * np.std(dists)
                    inlier_mask = dists < radius_threshold
                    pc = pc[inlier_mask]
                elif self.outlier_method == 'dbscan':
                    eps = self.kwargs.get('eps', 0.5)
                    min_samples = self.kwargs.get('min_samples', 10)
                    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pc[:, :3])
                    labels = clustering.labels_
                    inlier_mask = labels != -1
                    pc = pc[inlier_mask]
                elif self.outlier_method == 'box':
                    x_min, x_max = self.kwargs.get('x_range', (-np.inf, np.inf))
                    y_min, y_max = self.kwargs.get('y_range', (-np.inf, np.inf))
                    z_min, z_max = self.kwargs.get('z_range', (-np.inf, np.inf))
                    inlier_mask = (pc[:, 0] >= x_min) & (pc[:, 0] <= x_max) & \
                                  (pc[:, 1] >= y_min) & (pc[:, 1] <= y_max) & \
                                  (pc[:, 2] >= z_min) & (pc[:, 2] <= z_max)
                    pc = pc[inlier_mask]
                else:
                    raise ValueError(f"Unknown outlier_method: {self.outlier_method}")
                filtered_pc_seq.append(pc)

            results[key] = filtered_pc_seq

        return results
