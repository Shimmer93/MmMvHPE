"""Camera parameter prediction metrics.

Metrics for evaluating camera parameter predictions including:
- Translation error (Euclidean distance)
- Rotation error (geodesic distance in degrees)
- Focal length / Field of View error
"""

import numpy as np
import torch

from misc.utils import torch2numpy as to_numpy


def rotation_error_from_quats(pred_quats, gt_quats):
    """
    Compute geodesic rotation error between predicted and ground truth quaternions.
    
    Args:
        pred_quats: Predicted quaternions with shape (N, 4), format XYZW (scalar-last)
        gt_quats: Ground truth quaternions with shape (N, 4), format XYZW (scalar-last)
    
    Returns:
        Rotation errors in degrees with shape (N,)
    """
    # Normalize quaternions to ensure unit quaternions
    pred_quats = pred_quats / (np.linalg.norm(pred_quats, axis=-1, keepdims=True) + 1e-8)
    gt_quats = gt_quats / (np.linalg.norm(gt_quats, axis=-1, keepdims=True) + 1e-8)
    
    # Compute the dot product (cosine of half the angle)
    # Handle quaternion double cover: q and -q represent the same rotation
    dot_product = np.sum(pred_quats * gt_quats, axis=-1)
    dot_product = np.abs(dot_product)  # Handle double cover
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Geodesic distance: angle = 2 * arccos(|q1 · q2|)
    angle_rad = 2.0 * np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def translation_error_func(pred_trans, gt_trans, reduce=True):
    """
    Compute Euclidean distance between predicted and ground truth translations.
    
    Args:
        pred_trans: Predicted translations with shape (N, 3)
        gt_trans: Ground truth translations with shape (N, 3)
        reduce: If True, return mean error; otherwise return per-sample errors
    
    Returns:
        Translation error(s) in the same unit as input (usually meters)
    """
    error = np.linalg.norm(pred_trans - gt_trans, axis=-1)
    if reduce:
        return np.mean(error)
    return error


def rotation_error_func(pred_quats, gt_quats, reduce=True):
    """
    Compute rotation error between predicted and ground truth quaternions.
    
    Args:
        pred_quats: Predicted quaternions with shape (N, 4), format XYZW
        gt_quats: Ground truth quaternions with shape (N, 4), format XYZW
        reduce: If True, return mean error; otherwise return per-sample errors
    
    Returns:
        Rotation error(s) in degrees
    """
    error = rotation_error_from_quats(pred_quats, gt_quats)
    if reduce:
        return np.mean(error)
    return error


def fov_error_func(pred_fov, gt_fov, reduce=True):
    """
    Compute field of view error between predicted and ground truth FoV.
    
    Args:
        pred_fov: Predicted FoV with shape (N, 2) for (fov_h, fov_w) in radians
        gt_fov: Ground truth FoV with shape (N, 2) in radians
        reduce: If True, return mean error; otherwise return per-sample errors
    
    Returns:
        FoV error(s) in degrees
    """
    # Convert to degrees for more interpretable error
    error = np.abs(pred_fov - gt_fov)
    error_deg = np.degrees(error)
    
    # Average over h and w dimensions
    error_deg = np.mean(error_deg, axis=-1)
    
    if reduce:
        return np.mean(error_deg)
    return error_deg


def focal_error_func(pred_fov, gt_fov, image_size_hw, reduce=True):
    """
    Compute focal length error by converting FoV to focal length.
    
    Args:
        pred_fov: Predicted FoV with shape (N, 2) for (fov_h, fov_w) in radians
        gt_fov: Ground truth FoV with shape (N, 2) in radians
        image_size_hw: Tuple of (height, width) of the image
        reduce: If True, return mean error; otherwise return per-sample errors
    
    Returns:
        Focal length error(s) in pixels (average of fx and fy errors)
    """
    H, W = image_size_hw
    
    # Convert FoV to focal length: f = (size/2) / tan(fov/2)
    pred_fy = (H / 2.0) / np.tan(pred_fov[..., 0] / 2.0 + 1e-8)
    pred_fx = (W / 2.0) / np.tan(pred_fov[..., 1] / 2.0 + 1e-8)
    
    gt_fy = (H / 2.0) / np.tan(gt_fov[..., 0] / 2.0 + 1e-8)
    gt_fx = (W / 2.0) / np.tan(gt_fov[..., 1] / 2.0 + 1e-8)
    
    error_fx = np.abs(pred_fx - gt_fx)
    error_fy = np.abs(pred_fy - gt_fy)
    
    # Average fx and fy errors
    error = (error_fx + error_fy) / 2.0
    
    if reduce:
        return np.mean(error)
    return error


class CameraTranslationError:
    """Metric for camera translation prediction error."""
    
    def __init__(self, modality=None):
        """
        Args:
            modality: Modality name (e.g., 'rgb', 'depth'). If None, uses default keys.
        """
        self.modality = modality
        if modality is not None:
            self.name = f'cam_trans_err_{modality}'
        else:
            self.name = 'cam_trans_err'
    
    def __call__(self, preds, targets):
        """
        Compute camera translation error.
        
        Args:
            preds: Dictionary containing 'pred_cameras' with list of camera encodings
            targets: Dictionary containing ground truth camera encodings
        
        Returns:
            Mean translation error in meters
        """
        if self.modality is not None:
            gt_key = f'gt_camera_{self.modality}'
        else:
            gt_key = 'gt_camera'
        
        pred_cameras = preds.get('pred_cameras')
        gt_cameras = targets.get(gt_key)
        
        if pred_cameras is None or gt_cameras is None:
            return 0.0
        
        # pred_cameras is a list from iterative refinement; use the last one
        if isinstance(pred_cameras, list):
            pred_cameras = pred_cameras[-1]
        
        pred_cameras = to_numpy(pred_cameras)
        gt_cameras = to_numpy(_select_gt_camera_last_frame(gt_cameras))
        
        # Handle batch dimensions - flatten if needed
        if pred_cameras.ndim > 2:
            pred_cameras = pred_cameras.reshape(-1, pred_cameras.shape[-1])
        if gt_cameras.ndim > 2:
            gt_cameras = gt_cameras.reshape(-1, gt_cameras.shape[-1])
        
        # Extract translation (first 3 components)
        pred_trans = pred_cameras[..., :3]
        gt_trans = gt_cameras[..., :3]
        
        return translation_error_func(pred_trans, gt_trans, reduce=True)


class CameraRotationError:
    """Metric for camera rotation prediction error."""
    
    def __init__(self, modality=None):
        """
        Args:
            modality: Modality name (e.g., 'rgb', 'depth'). If None, uses default keys.
        """
        self.modality = modality
        if modality is not None:
            self.name = f'cam_rot_err_{modality}'
        else:
            self.name = 'cam_rot_err'
    
    def __call__(self, preds, targets):
        """
        Compute camera rotation error.
        
        Args:
            preds: Dictionary containing 'pred_cameras' with list of camera encodings
            targets: Dictionary containing ground truth camera encodings
        
        Returns:
            Mean rotation error in degrees
        """
        if self.modality is not None:
            gt_key = f'gt_camera_{self.modality}'
        else:
            gt_key = 'gt_camera'
        
        pred_cameras = preds.get('pred_cameras')
        gt_cameras = targets.get(gt_key)
        
        if pred_cameras is None or gt_cameras is None:
            return 0.0
        
        # pred_cameras is a list from iterative refinement; use the last one
        if isinstance(pred_cameras, list):
            pred_cameras = pred_cameras[-1]
        
        pred_cameras = to_numpy(pred_cameras)
        gt_cameras = to_numpy(_select_gt_camera_last_frame(gt_cameras))
        
        # Handle batch dimensions - flatten if needed
        if pred_cameras.ndim > 2:
            pred_cameras = pred_cameras.reshape(-1, pred_cameras.shape[-1])
        if gt_cameras.ndim > 2:
            gt_cameras = gt_cameras.reshape(-1, gt_cameras.shape[-1])
        
        # Extract quaternion (components 3:7)
        pred_quats = pred_cameras[..., 3:7]
        gt_quats = gt_cameras[..., 3:7]
        
        return rotation_error_func(pred_quats, gt_quats, reduce=True)


class CameraFoVError:
    """Metric for camera field of view prediction error."""
    
    def __init__(self, modality=None):
        """
        Args:
            modality: Modality name (e.g., 'rgb', 'depth'). If None, uses default keys.
        """
        self.modality = modality
        if modality is not None:
            self.name = f'cam_fov_err_{modality}'
        else:
            self.name = 'cam_fov_err'
    
    def __call__(self, preds, targets):
        """
        Compute camera field of view error.
        
        Args:
            preds: Dictionary containing 'pred_cameras' with list of camera encodings
            targets: Dictionary containing ground truth camera encodings
        
        Returns:
            Mean FoV error in degrees
        """
        if self.modality is not None:
            gt_key = f'gt_camera_{self.modality}'
        else:
            gt_key = 'gt_camera'
        
        pred_cameras = preds.get('pred_cameras')
        gt_cameras = targets.get(gt_key)
        
        if pred_cameras is None or gt_cameras is None:
            return 0.0
        
        # pred_cameras is a list from iterative refinement; use the last one
        if isinstance(pred_cameras, list):
            pred_cameras = pred_cameras[-1]
        
        pred_cameras = to_numpy(pred_cameras)
        gt_cameras = to_numpy(_select_gt_camera_last_frame(gt_cameras))
        
        # Handle batch dimensions - flatten if needed
        if pred_cameras.ndim > 2:
            pred_cameras = pred_cameras.reshape(-1, pred_cameras.shape[-1])
        if gt_cameras.ndim > 2:
            gt_cameras = gt_cameras.reshape(-1, gt_cameras.shape[-1])
        
        # Extract FoV (components 7:9)
        pred_fov = pred_cameras[..., 7:9]
        gt_fov = gt_cameras[..., 7:9]
        
        return fov_error_func(pred_fov, gt_fov, reduce=True)


class CameraFocalError:
    """Metric for camera focal length prediction error."""
    
    def __init__(self, modality=None, image_size_hw=(256, 256)):
        """
        Args:
            modality: Modality name (e.g., 'rgb', 'depth'). If None, uses default keys.
            image_size_hw: Tuple of (height, width) for focal length computation.
        """
        self.modality = modality
        self.image_size_hw = image_size_hw
        if modality is not None:
            self.name = f'cam_focal_err_{modality}'
        else:
            self.name = 'cam_focal_err'
    
    def __call__(self, preds, targets):
        """
        Compute camera focal length error.
        
        Args:
            preds: Dictionary containing 'pred_cameras' with list of camera encodings
            targets: Dictionary containing ground truth camera encodings
        
        Returns:
            Mean focal length error in pixels
        """
        if self.modality is not None:
            gt_key = f'gt_camera_{self.modality}'
        else:
            gt_key = 'gt_camera'
        
        pred_cameras = preds.get('pred_cameras')
        gt_cameras = targets.get(gt_key)
        
        if pred_cameras is None or gt_cameras is None:
            return 0.0
        
        # pred_cameras is a list from iterative refinement; use the last one
        if isinstance(pred_cameras, list):
            pred_cameras = pred_cameras[-1]
        
        pred_cameras = to_numpy(pred_cameras)
        gt_cameras = to_numpy(_select_gt_camera_last_frame(gt_cameras))
        
        # Handle batch dimensions - flatten if needed
        if pred_cameras.ndim > 2:
            pred_cameras = pred_cameras.reshape(-1, pred_cameras.shape[-1])
        if gt_cameras.ndim > 2:
            gt_cameras = gt_cameras.reshape(-1, gt_cameras.shape[-1])
        
        # Extract FoV (components 7:9)
        pred_fov = pred_cameras[..., 7:9]
        gt_fov = gt_cameras[..., 7:9]
        
        return focal_error_func(pred_fov, gt_fov, self.image_size_hw, reduce=True)


class CameraTranslationL2Error:
    """Metric for mean L2 translation error."""

    def __init__(self, modality=None):
        self.modality = modality
        if modality is not None:
            self.name = f'cam_trans_l2_{modality}'
        else:
            self.name = 'cam_trans_l2'

    def __call__(self, preds, targets):
        if self.modality is not None:
            gt_key = f'gt_camera_{self.modality}'
        else:
            gt_key = 'gt_camera'

        pred_cameras = preds.get('pred_cameras')
        gt_cameras = targets.get(gt_key)

        if pred_cameras is None or gt_cameras is None:
            return 0.0

        pred_cameras = self._select_modality(pred_cameras, targets)
        gt_cameras = self._select_gt_camera(gt_cameras)
        if pred_cameras is None or gt_cameras is None:
            return 0.0

        pred_cameras = to_numpy(pred_cameras)
        gt_cameras = to_numpy(gt_cameras)

        if pred_cameras.ndim > 2:
            pred_cameras = pred_cameras.reshape(-1, pred_cameras.shape[-1])
        if gt_cameras.ndim > 2:
            gt_cameras = gt_cameras.reshape(-1, gt_cameras.shape[-1])

        pred_trans = pred_cameras[..., :3]
        gt_trans = gt_cameras[..., :3]

        return translation_error_func(pred_trans, gt_trans, reduce=True)

    def _select_modality(self, pred_cameras, targets):
        if pred_cameras is None:
            return None
        if isinstance(pred_cameras, list):
            pred_cameras = pred_cameras[-1]
        if pred_cameras.ndim != 3 or self.modality is None:
            return pred_cameras
        modalities = targets.get('modalities', [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]
        if not modalities:
            modalities = [
                mod for mod in ["rgb", "depth", "lidar", "mmwave"]
                if f"gt_camera_{mod}" in targets
            ]
        if not modalities or self.modality not in modalities:
            return None
        return pred_cameras[:, modalities.index(self.modality)]

    @staticmethod
    def _select_gt_camera(gt_cameras):
        return _select_gt_camera_last_frame(gt_cameras)


class CameraRotationAngleError:
    """Metric for mean rotation angle error in degrees."""

    def __init__(self, modality=None):
        self.modality = modality
        if modality is not None:
            self.name = f'cam_rot_angle_{modality}'
        else:
            self.name = 'cam_rot_angle'

    def __call__(self, preds, targets):
        if self.modality is not None:
            gt_key = f'gt_camera_{self.modality}'
        else:
            gt_key = 'gt_camera'

        pred_cameras = preds.get('pred_cameras')
        gt_cameras = targets.get(gt_key)

        if pred_cameras is None or gt_cameras is None:
            return 0.0

        pred_cameras = self._select_modality(pred_cameras, targets)
        gt_cameras = self._select_gt_camera(gt_cameras)
        if pred_cameras is None or gt_cameras is None:
            return 0.0

        pred_cameras = to_numpy(pred_cameras)
        gt_cameras = to_numpy(gt_cameras)

        if pred_cameras.ndim > 2:
            pred_cameras = pred_cameras.reshape(-1, pred_cameras.shape[-1])
        if gt_cameras.ndim > 2:
            gt_cameras = gt_cameras.reshape(-1, gt_cameras.shape[-1])

        pred_quats = pred_cameras[..., 3:7]
        gt_quats = gt_cameras[..., 3:7]

        return rotation_error_func(pred_quats, gt_quats, reduce=True)

    def _select_modality(self, pred_cameras, targets):
        if pred_cameras is None:
            return None
        if isinstance(pred_cameras, list):
            pred_cameras = pred_cameras[-1]
        if pred_cameras.ndim != 3 or self.modality is None:
            return pred_cameras
        modalities = targets.get('modalities', [])
        if modalities and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]
        if not modalities:
            modalities = [
                mod for mod in ["rgb", "depth", "lidar", "mmwave"]
                if f"gt_camera_{mod}" in targets
            ]
        if not modalities or self.modality not in modalities:
            return None
        return pred_cameras[:, modalities.index(self.modality)]

    @staticmethod
    def _select_gt_camera(gt_cameras):
        return _select_gt_camera_last_frame(gt_cameras)


def _rotation_angle_from_quats(pred_quats, gt_quats, eps=1e-15):
    pred_quats = pred_quats / (np.linalg.norm(pred_quats, axis=-1, keepdims=True) + eps)
    gt_quats = gt_quats / (np.linalg.norm(gt_quats, axis=-1, keepdims=True) + eps)
    dot = np.sum(pred_quats * gt_quats, axis=-1)
    loss_q = np.clip(1.0 - dot**2, eps, None)
    err_q = np.arccos(1.0 - 2.0 * loss_q)
    return np.degrees(err_q)


def _translation_angle(pred_t, gt_t, eps=1e-15, ambiguity=True, default_err=1e6):
    pred_norm = np.linalg.norm(pred_t, axis=-1, keepdims=True)
    gt_norm = np.linalg.norm(gt_t, axis=-1, keepdims=True)
    pred_unit = pred_t / (pred_norm + eps)
    gt_unit = gt_t / (gt_norm + eps)
    loss_t = np.clip(1.0 - np.sum(pred_unit * gt_unit, axis=-1) ** 2, eps, None)
    err_t = np.arccos(np.sqrt(1.0 - loss_t))
    err_t = np.where(np.isfinite(err_t), err_t, default_err)
    err_deg = np.degrees(err_t)
    if ambiguity:
        err_deg = np.minimum(err_deg, np.abs(180.0 - err_deg))
    return err_deg


def _calculate_auc_np(r_error, t_error, max_threshold=30):
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / max(num_pairs, 1.0)
    return np.mean(np.cumsum(normalized_histogram))


def _select_gt_camera_last_frame(gt_cameras):
    if gt_cameras is None:
        return None
    if isinstance(gt_cameras, np.ndarray):
        if gt_cameras.ndim == 4:
            return gt_cameras[:, :, -1, :].mean(axis=1)
        if gt_cameras.ndim == 3:
            return gt_cameras[:, -1, :]
        return gt_cameras
    if isinstance(gt_cameras, torch.Tensor):
        if gt_cameras.dim() == 4:
            return gt_cameras[:, :, -1, :].mean(dim=1)
        if gt_cameras.dim() == 3:
            return gt_cameras[:, -1, :]
        return gt_cameras
    return gt_cameras


class CameraPoseAUC:
    """AUC metric for absolute camera pose errors in the pelvis-based human frame."""

    def __init__(self, modality, max_threshold=30):
        self.modality = modality
        self.max_threshold = max_threshold
        self.name = f'cam_pose_auc_{max_threshold}_{modality}'

    def __call__(self, preds, targets):
        pred_cameras = preds.get('pred_cameras')
        if pred_cameras is None:
            return 0.0

        if isinstance(pred_cameras, list):
            pred_cameras = pred_cameras[-1]

        pred_cameras = to_numpy(pred_cameras)
        gt_cameras = targets.get(f'gt_camera_{self.modality}')
        if gt_cameras is None:
            return 0.0

        gt_cameras = _select_gt_camera_last_frame(gt_cameras)
        gt_cameras = to_numpy(gt_cameras)

        if pred_cameras.ndim == 3:
            modalities = targets.get('modalities', [])
            if modalities and isinstance(modalities[0], (list, tuple)):
                modalities = modalities[0]
            if not modalities:
                modalities = [
                    mod for mod in ["rgb", "depth", "lidar", "mmwave"]
                    if f"gt_camera_{mod}" in targets
                ]
            if not modalities:
                return 0.0
            if self.modality not in modalities:
                return 0.0
            modality_idx = modalities.index(self.modality)
            pred_cameras = pred_cameras[:, modality_idx]

        if pred_cameras.ndim > 2:
            pred_cameras = pred_cameras.reshape(-1, pred_cameras.shape[-1])
        if gt_cameras.ndim > 2:
            gt_cameras = gt_cameras.reshape(-1, gt_cameras.shape[-1])

        pred_quats = pred_cameras[..., 3:7]
        gt_quats = gt_cameras[..., 3:7]
        pred_trans = pred_cameras[..., :3]
        gt_trans = gt_cameras[..., :3]

        r_error = _rotation_angle_from_quats(pred_quats, gt_quats)
        t_error = _translation_angle(pred_trans, gt_trans)

        return _calculate_auc_np(r_error, t_error, max_threshold=self.max_threshold)
