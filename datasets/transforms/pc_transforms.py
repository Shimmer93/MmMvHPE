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
    ):
        assert center_type in ['mean', 'median'], "center_type must be 'mean' or 'median'"
        self.center_type = center_type
        self.keys = keys
        self.keypoints_key = keypoints_key
        self.shifted_keypoints_suffix = shifted_keypoints_suffix

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
    def _is_pelvis_centered(keypoints):
        if isinstance(keypoints, np.ndarray):
            if keypoints.shape[-1] != 3 or keypoints.size < 3:
                return False
            return np.linalg.norm(keypoints.reshape(-1, 3)[0]) < 1e-4
        if isinstance(keypoints, torch.Tensor):
            if keypoints.shape[-1] != 3 or keypoints.numel() < 3:
                return False
            return torch.linalg.norm(keypoints.reshape(-1, 3)[0]).item() < 1e-4
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
                        if self._is_pelvis_centered(keypoints):
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
