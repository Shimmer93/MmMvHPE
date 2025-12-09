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

            if f'{key}_camera' in sample:
                camera = deepcopy(sample[f'{key}_camera'])
                camera['intrinsic'][0, 0] = camera['intrinsic'][0, 0] * (new_w / w)
                camera['intrinsic'][1, 1] = camera['intrinsic'][1, 1] * (new_h / h)
                camera['intrinsic'][0, 2] = camera['intrinsic'][0, 2] * (new_w / w) + pad_w // 2
                camera['intrinsic'][1, 2] = camera['intrinsic'][1, 2] * (new_h / h) + pad_h // 2
                sample[f'{key}_camera'] = camera

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
    

