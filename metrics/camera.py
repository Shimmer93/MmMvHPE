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
        gt_cameras = to_numpy(gt_cameras)
        
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
        gt_cameras = to_numpy(gt_cameras)
        
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
        gt_cameras = to_numpy(gt_cameras)
        
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
        gt_cameras = to_numpy(gt_cameras)
        
        # Handle batch dimensions - flatten if needed
        if pred_cameras.ndim > 2:
            pred_cameras = pred_cameras.reshape(-1, pred_cameras.shape[-1])
        if gt_cameras.ndim > 2:
            gt_cameras = gt_cameras.reshape(-1, gt_cameras.shape[-1])
        
        # Extract FoV (components 7:9)
        pred_fov = pred_cameras[..., 7:9]
        gt_fov = gt_cameras[..., 7:9]
        
        return focal_error_func(pred_fov, gt_fov, self.image_size_hw, reduce=True)
