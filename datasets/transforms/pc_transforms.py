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
            padded_pc_seq = []

            for pc in pc_seq:
                N, C = pc.shape

                if N == 0:
                    pc = np.random.normal(0, 1, (self.num_points, C)).astype(pc.dtype)

                elif N < self.num_points:
                    if self.pad_mode == 'zero':
                        pad = np.zeros((self.num_points - N, C), dtype=pc.dtype)
                    elif self.pad_mode == 'repeat':
                        np.random.shuffle(pc)
                        repeat_times = (self.num_points - N) // N + 1
                        pad = np.tile(pc, (repeat_times, 1))[:self.num_points - N]
                    pc = np.vstack((pc, pad))

                elif N > self.num_points:
                    np.random.shuffle(pc)
                    pc = pc[:self.num_points]

                padded_pc_seq.append(pc)

            results[key] = padded_pc_seq

        return results
    
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

    def __call__(self, results):
        for key in self.keys:
            if key not in results:
                results[f'{key}_affine'] = initialize_affine_matrix()
                continue
            pc_seq = results[key]
            all_ps = np.concatenate(pc_seq, axis=0)
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
            camera = results.get(camera_key)
            extrinsic_before = None
            if camera is not None and "extrinsic" in camera:
                extrinsic_before = np.asarray(camera["extrinsic"], dtype=np.float32)

            centered_pc_seq = []
            for pc in pc_seq:
                pc[:, :3] = pc[:, :3] - center
                centered_pc_seq.append(pc)
            results[key] = centered_pc_seq
            if f'{key}_affine' not in results:
                results[f'{key}_affine'] = initialize_affine_matrix()
            results[f'{key}_affine'][:3, 3] = results[f'{key}_affine'][:3, 3] - center

            if self.keypoints_key in results and results[self.keypoints_key] is not None:
                keypoints = results[self.keypoints_key]
                keypoints_for_pc = keypoints
                if extrinsic_before is not None:
                    if isinstance(keypoints, np.ndarray):
                        if keypoints.shape[-1] == 3 and keypoints.shape[0] >= 1:
                            pelvis_norm = np.linalg.norm(keypoints.reshape(-1, 3)[0])
                            if pelvis_norm < 1e-4:
                                keypoints_for_pc = self._apply_extrinsic(keypoints, extrinsic_before)
                    elif isinstance(keypoints, torch.Tensor):
                        if keypoints.shape[-1] == 3 and keypoints.numel() >= 3:
                            pelvis_norm = torch.linalg.norm(keypoints.reshape(-1, 3)[0]).item()
                            if pelvis_norm < 1e-4:
                                keypoints_for_pc = self._apply_extrinsic(keypoints, extrinsic_before)

                if isinstance(keypoints_for_pc, torch.Tensor):
                    shifted = keypoints_for_pc.clone()
                    shifted[..., :3] = shifted[..., :3] - torch.as_tensor(center, dtype=shifted.dtype, device=shifted.device)
                else:
                    shifted = np.asarray(keypoints_for_pc, dtype=np.float32).copy()
                    shifted[..., :3] = shifted[..., :3] - center
                results[f'{self.keypoints_key}{self.shifted_keypoints_suffix}_{key}'] = shifted

            if extrinsic_before is not None:
                extrinsic_after = extrinsic_before.copy()
                extrinsic_after[:, 3] = extrinsic_after[:, 3] - center
                camera = dict(camera)
                camera["extrinsic"] = extrinsic_after.astype(np.float32)
                results[camera_key] = camera

                if gt_camera_key in results and results[gt_camera_key] is not None:
                    gt_camera = results[gt_camera_key]
                    if isinstance(gt_camera, torch.Tensor):
                        gt_cam = gt_camera.clone()
                        if gt_cam.dim() == 1:
                            gt_cam = gt_cam.unsqueeze(0)
                            squeeze_back = True
                        else:
                            squeeze_back = False
                        gt_cam[..., :3] = torch.as_tensor(
                            extrinsic_after[:, 3], dtype=gt_cam.dtype, device=gt_cam.device
                        )
                        results[gt_camera_key] = gt_cam.squeeze(0) if squeeze_back else gt_cam
                    else:
                        gt_cam = np.asarray(gt_camera, dtype=np.float32).copy()
                        if gt_cam.ndim == 1:
                            gt_cam = gt_cam[None, :]
                            squeeze_back = True
                        else:
                            squeeze_back = False
                        gt_cam[..., :3] = extrinsic_after[:, 3]
                        results[gt_camera_key] = gt_cam.squeeze(0) if squeeze_back else gt_cam

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
