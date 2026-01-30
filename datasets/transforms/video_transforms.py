import numpy as np
# import torch
from copy import deepcopy
from typing import Optional, Sequence, Union, List
import cv2

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]

class VideoResize():
    def __init__(self, 
                 size: Union[int, Sequence[int]],
                 keep_ratio: bool = True,
                 divided_by: Optional[int] = None,
                 interpolation: Optional[str] = 'bilinear',
                 keys: List[str] = ['input_rgb', 'input_depth']):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.keep_ratio = keep_ratio
        self.divided_by = divided_by
        self.interpolation = interpolation
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            frames = sample[key]
            resized_frames = []

            h, w = frames[0].shape[:2]
            if self.keep_ratio:
                scale = min(self.size[0] / h, self.size[1] / w)
                new_w, new_h = int(w * scale), int(h * scale)
            else:
                new_w, new_h = self.size[1], self.size[0]

            if self.divided_by is not None:
                pad_w = (self.divided_by - new_w % self.divided_by) % self.divided_by
                pad_h = (self.divided_by - new_h % self.divided_by) % self.divided_by
            else:
                pad_w, pad_h = 0, 0

            if self.interpolation == 'bilinear':
                interp_method = cv2.INTER_LINEAR
            elif self.interpolation == 'nearest':
                interp_method = cv2.INTER_NEAREST
            else:
                raise ValueError(f"Unsupported interpolation method: {self.interpolation}")
            
            for frame in frames:
                
                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=interp_method)
                if pad_w > 0 or pad_h > 0:
                    resized_frame = cv2.copyMakeBorder(resized_frame, pad_h // 2, pad_h - pad_h // 2,
                                                       pad_w // 2, pad_w - pad_w // 2,
                                                       borderType=cv2.BORDER_CONSTANT, value=0)
                resized_frames.append(resized_frame)

            # Update camera intrinsics if present
            # Handle both 'input_X_camera' and 'X_camera' naming conventions
            camera_key = f'{key}_camera'
            if camera_key not in sample:
                # Try without 'input_' prefix (e.g., 'rgb_camera' instead of 'input_rgb_camera')
                if key.startswith('input_'):
                    camera_key = f'{key[6:]}_camera'  # Remove 'input_' prefix
            
            if camera_key in sample:
                camera = deepcopy(sample[camera_key])
                camera['intrinsic'][0, 0] = camera['intrinsic'][0, 0] * (new_w / w)
                camera['intrinsic'][1, 1] = camera['intrinsic'][1, 1] * (new_h / h)
                camera['intrinsic'][0, 2] = camera['intrinsic'][0, 2] * (new_w / w) + pad_w // 2
                camera['intrinsic'][1, 2] = camera['intrinsic'][1, 2] * (new_h / h) + pad_h // 2
                sample[camera_key] = camera

            sample[key] = resized_frames
        return sample
                
class VideoNormalize():
    def __init__(self, 
                 norm_mode: str = 'imagenet',
                 mean: Sequence[float] = [123.675, 116.28, 103.53],
                 std: Sequence[float] = [58.395, 57.12, 57.375],
                 keys: List[str] = ['input_rgb']):
        
        assert norm_mode in ['imagenet', 'zero_one', 'custom'], "norm_mode should be 'imagenet', 'zero_one' or 'custom'"
        self.norm_mode = norm_mode

        if norm_mode == 'imagenet':
            self.mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
            self.std = np.array(IMAGENET_STD).reshape(1, 1, 3)
        elif norm_mode == 'zero_one':
            self.mean = np.array([0.0, 0.0, 0.0]).reshape(1, 1, 3)
            self.std = np.array([255.0, 255.0, 255.0]).reshape(1, 1, 3)
        else:
            self.mean = np.array(mean).reshape(1, 1, 3)
            self.std = np.array(std).reshape(1, 1, 3)
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            frames = sample[key]
            normalized_frames = []
            for frame in frames:
                # Handle both RGB (H, W, 3) and grayscale (H, W) images
                if frame.ndim == 2:
                    # Grayscale image - use scalar mean/std
                    mean = self.mean[0, 0, 0]
                    std = self.std[0, 0, 0]
                    frame = (frame.astype(np.float32) - mean) / std
                else:
                    # RGB image - use channel-wise mean/std
                    frame = (frame.astype(np.float32) - self.mean) / self.std
                normalized_frames.append(frame)
            sample[key] = normalized_frames
        return sample


class VideoRandomCropOrPad():
    def __init__(
        self,
        scale_range: Sequence[float],
        keys: List[str] = ['input_rgb', 'input_depth'],
        pad_value: Union[int, float, Sequence[int]] = 0,
    ):
        if len(scale_range) != 2:
            raise ValueError("scale_range must be a 2-element sequence.")
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        if not (self.scale_min < 1.0 and self.scale_max > 1.0):
            raise ValueError("scale_range must satisfy a < 1 and b > 1.")
        self.keys = keys
        self.pad_value = pad_value

    def __call__(self, sample):
        scale = float(np.random.uniform(self.scale_min, self.scale_max))
        if abs(scale - 1.0) < 1e-6:
            return sample

        for key in self.keys:
            frames = sample[key]
            if not frames:
                continue

            h, w = frames[0].shape[:2]
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            if scale < 1.0:
                max_x0 = w - new_w
                max_y0 = h - new_h
                x0 = int(np.random.randint(0, max_x0 + 1)) if max_x0 > 0 else 0
                y0 = int(np.random.randint(0, max_y0 + 1)) if max_y0 > 0 else 0
                x1 = x0 + new_w
                y1 = y0 + new_h
                processed_frames = [frame[y0:y1, x0:x1] for frame in frames]
                shift_x = -x0
                shift_y = -y0
            else:
                pad_w = new_w - w
                pad_h = new_h - h
                pad_left = int(np.random.randint(0, pad_w + 1)) if pad_w > 0 else 0
                pad_top = int(np.random.randint(0, pad_h + 1)) if pad_h > 0 else 0
                pad_right = pad_w - pad_left
                pad_bottom = pad_h - pad_top

                processed_frames = [
                    cv2.copyMakeBorder(
                        frame,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        borderType=cv2.BORDER_CONSTANT,
                        value=self.pad_value,
                    )
                    for frame in frames
                ]
                shift_x = pad_left
                shift_y = pad_top

            camera_key = f'{key}_camera'
            if camera_key not in sample and key.startswith('input_'):
                camera_key = f'{key[6:]}_camera'

            if camera_key in sample:
                camera = deepcopy(sample[camera_key])
                camera['intrinsic'][0, 2] = camera['intrinsic'][0, 2] + shift_x
                camera['intrinsic'][1, 2] = camera['intrinsic'][1, 2] + shift_y
                sample[camera_key] = camera

            sample[key] = processed_frames

        return sample


class VideoRandomRotate():
    def __init__(
        self,
        angle_range: Sequence[float],
        keys: List[str] = ['input_rgb', 'input_depth'],
        pad_value: Union[int, float, Sequence[int]] = 0,
        interpolation: str = 'bilinear',
    ):
        if len(angle_range) != 2:
            raise ValueError("angle_range must be a 2-element sequence.")
        self.angle_min = float(angle_range[0])
        self.angle_max = float(angle_range[1])
        self.keys = keys
        self.pad_value = pad_value
        self.interpolation = interpolation

    def __call__(self, sample):
        angle = float(np.random.uniform(self.angle_min, self.angle_max))
        if abs(angle) < 1e-6:
            return sample

        if self.interpolation == 'bilinear':
            interp_method = cv2.INTER_LINEAR
        elif self.interpolation == 'nearest':
            interp_method = cv2.INTER_NEAREST
        else:
            raise ValueError(f"Unsupported interpolation method: {self.interpolation}")

        for key in self.keys:
            frames = sample[key]
            if not frames:
                continue

            h, w = frames[0].shape[:2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            rotated_frames = [
                cv2.warpAffine(
                    frame,
                    M,
                    (w, h),
                    flags=interp_method,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.pad_value,
                )
                for frame in frames
            ]

            camera_key = f'{key}_camera'
            if camera_key not in sample and key.startswith('input_'):
                camera_key = f'{key[6:]}_camera'

            if camera_key in sample:
                camera = deepcopy(sample[camera_key])
                K = camera['intrinsic']
                if isinstance(K, np.ndarray):
                    K_mat = K.astype(np.float32)
                else:
                    K_mat = np.array(K, dtype=np.float32)

                H = np.array(
                    [
                        [M[0, 0], M[0, 1], M[0, 2]],
                        [M[1, 0], M[1, 1], M[1, 2]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
                K_new = H @ K_mat
                camera['intrinsic'] = K_new
                sample[camera_key] = camera

            sample[key] = rotated_frames

        return sample


class VideoNormalizeFoV():
    def __init__(
        self,
        target_fov: Optional[Sequence[float]] = None,
        target_focal: Optional[Sequence[float]] = None,
        keys: List[str] = ['input_rgb'],
        interpolation: str = 'bilinear',
    ):
        if target_fov is None and target_focal is None:
            raise ValueError("Provide either target_fov or target_focal.")
        if target_fov is not None and target_focal is not None:
            raise ValueError("Provide only one of target_fov or target_focal.")
        self.target_fov = target_fov
        self.target_focal = target_focal
        self.keys = keys
        self.interpolation = interpolation

    def __call__(self, sample):
        if self.interpolation == 'bilinear':
            interp_method = cv2.INTER_LINEAR
        elif self.interpolation == 'nearest':
            interp_method = cv2.INTER_NEAREST
        else:
            raise ValueError(f"Unsupported interpolation method: {self.interpolation}")

        for key in self.keys:
            frames = sample[key]
            if not frames:
                continue

            camera_key = f'{key}_camera'
            if camera_key not in sample and key.startswith('input_'):
                camera_key = f'{key[6:]}_camera'
            if camera_key not in sample:
                continue

            camera = deepcopy(sample[camera_key])
            K_src = np.array(camera['intrinsic'], dtype=np.float32)
            h, w = frames[0].shape[:2]

            K_can = K_src.copy()
            if self.target_focal is not None:
                fx, fy = self.target_focal
                K_can[0, 0] = float(fx)
                K_can[1, 1] = float(fy)
                K_can[0, 2] = w / 2.0
                K_can[1, 2] = h / 2.0
            else:
                fov_h, fov_w = self.target_fov
                fy = (h / 2.0) / np.tan(float(fov_h) / 2.0)
                fx = (w / 2.0) / np.tan(float(fov_w) / 2.0)
                K_can[0, 0] = fx
                K_can[1, 1] = fy
                K_can[0, 2] = w / 2.0
                K_can[1, 2] = h / 2.0

            H = K_can @ np.linalg.inv(K_src)

            warped_frames = [
                cv2.warpPerspective(frame, H, (w, h), flags=interp_method, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                for frame in frames
            ]

            camera['intrinsic'] = K_can
            sample[camera_key] = camera
            sample[key] = warped_frames

        return sample
    
