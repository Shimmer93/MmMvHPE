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


class VideoCenterCropResize():
    def __init__(
        self,
        size: Sequence[int],
        interpolation: str = 'bilinear',
        keys: List[str] = ['input_rgb'],
        update_2d_keypoints: bool = True,
    ):
        if len(size) != 2:
            raise ValueError("size must be a 2-element sequence [H, W].")
        self.size = (int(size[0]), int(size[1]))
        self.interpolation = interpolation
        self.keys = keys
        self.update_2d_keypoints = bool(update_2d_keypoints)

    @staticmethod
    def _camera_key_from_input_key(key: str) -> str:
        if key.startswith('input_'):
            return f'{key[6:]}_camera'
        return f'{key}_camera'

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
    def _is_multiview_frames(frames):
        return (
            isinstance(frames, (list, tuple))
            and len(frames) > 0
            and isinstance(frames[0], (list, tuple))
        )

    @staticmethod
    def _denormalize_points_2d(points_2d, image_size_hw):
        h, w = int(image_size_hw[0]), int(image_size_hw[1])
        pts = np.asarray(points_2d, dtype=np.float32).copy()
        pts[..., 0] = (pts[..., 0] + 1.0) * 0.5 * max(w - 1, 1)
        pts[..., 1] = (pts[..., 1] + 1.0) * 0.5 * max(h - 1, 1)
        return pts

    @staticmethod
    def _normalize_points_2d(points_2d, image_size_hw):
        h, w = int(image_size_hw[0]), int(image_size_hw[1])
        pts = np.asarray(points_2d, dtype=np.float32).copy()
        pts[..., 0] = 2.0 * (pts[..., 0] / max(w - 1, 1)) - 1.0
        pts[..., 1] = 2.0 * (pts[..., 1] / max(h - 1, 1)) - 1.0
        return pts

    @staticmethod
    def _crop_box_for_aspect(h: int, w: int, target_h: int, target_w: int):
        target_aspect = float(target_h) / float(target_w)
        current_aspect = float(h) / float(w)
        if abs(current_aspect - target_aspect) < 1e-6:
            return 0, 0, w, h
        if current_aspect > target_aspect:
            crop_h = max(1, int(round(w * target_aspect)))
            crop_w = w
            x0 = 0
            y0 = max(0, (h - crop_h) // 2)
        else:
            crop_h = h
            crop_w = max(1, int(round(h / target_aspect)))
            x0 = max(0, (w - crop_w) // 2)
            y0 = 0
        return x0, y0, crop_w, crop_h

    def _resize_frame(self, frame, out_w, out_h):
        if self.interpolation == 'bilinear':
            interp_method = cv2.INTER_LINEAR
        elif self.interpolation == 'nearest':
            interp_method = cv2.INTER_NEAREST
        else:
            raise ValueError(f"Unsupported interpolation method: {self.interpolation}")
        return cv2.resize(frame, (out_w, out_h), interpolation=interp_method)

    def _transform_camera(self, camera, x0, y0, crop_w, crop_h, out_w, out_h):
        if camera is None:
            return None
        cam = deepcopy(camera)
        intrinsic = np.asarray(cam['intrinsic'], dtype=np.float32).copy()
        intrinsic[0, 2] -= float(x0)
        intrinsic[1, 2] -= float(y0)
        scale_x = float(out_w) / float(crop_w)
        scale_y = float(out_h) / float(crop_h)
        intrinsic[0, 0] *= scale_x
        intrinsic[1, 1] *= scale_y
        intrinsic[0, 2] *= scale_x
        intrinsic[1, 2] *= scale_y
        cam['intrinsic'] = intrinsic
        return cam

    def _transform_points_2d(self, points_2d, in_hw, crop_box):
        in_h, in_w = in_hw
        x0, y0, crop_w, crop_h = crop_box
        out_h, out_w = self.size
        pts = np.asarray(points_2d, dtype=np.float32)
        if pts.size == 0:
            return pts.astype(np.float32)
        is_normalized = np.isfinite(pts).all() and float(np.nanmax(np.abs(pts))) <= 1.5
        if is_normalized:
            pts = self._denormalize_points_2d(pts, (in_h, in_w))
        pts = pts.copy()
        pts[..., 0] = (pts[..., 0] - float(x0)) * (float(out_w) / float(crop_w))
        pts[..., 1] = (pts[..., 1] - float(y0)) * (float(out_h) / float(crop_h))
        if is_normalized:
            pts = self._normalize_points_2d(pts, (out_h, out_w))
        return pts.astype(np.float32)

    def __call__(self, sample):
        for key in self.keys:
            if key not in sample:
                continue
            frames = sample[key]
            if not frames:
                continue

            out_h, out_w = self.size
            camera_key = self._camera_key_from_input_key(key)
            camera_raw = sample.get(camera_key)

            if self._is_multiview_frames(frames):
                transformed_views = []
                camera_list = self._to_camera_list(camera_raw)
                if camera_list and len(camera_list) != len(frames):
                    raise ValueError(
                        f"Camera/view count mismatch for {key}: {len(camera_list)} cameras vs {len(frames)} views."
                    )
                updated_cameras = []
                crop_boxes = []
                input_shapes = []
                for view_idx, view_frames in enumerate(frames):
                    if not view_frames:
                        transformed_views.append(view_frames)
                        continue
                    h, w = view_frames[0].shape[:2]
                    crop_box = self._crop_box_for_aspect(h, w, out_h, out_w)
                    x0, y0, crop_w, crop_h = crop_box
                    transformed_views.append(
                        [
                            self._resize_frame(frame[y0:y0 + crop_h, x0:x0 + crop_w], out_w, out_h)
                            for frame in view_frames
                        ]
                    )
                    crop_boxes.append(crop_box)
                    input_shapes.append((h, w))
                    if camera_list:
                        updated_cameras.append(
                            self._transform_camera(camera_list[view_idx], x0, y0, crop_w, crop_h, out_w, out_h)
                        )
                sample[key] = transformed_views
                if updated_cameras:
                    sample[camera_key] = self._restore_camera_type(updated_cameras, camera_raw)

                if (
                    self.update_2d_keypoints
                    and key == 'input_rgb'
                    and 'gt_keypoints_2d_rgb' in sample
                    and crop_boxes
                ):
                    kpts = np.asarray(sample['gt_keypoints_2d_rgb'], dtype=np.float32)
                    if kpts.ndim >= 3 and kpts.shape[0] == len(crop_boxes):
                        out = []
                        for view_idx, crop_box in enumerate(crop_boxes):
                            out.append(self._transform_points_2d(kpts[view_idx], input_shapes[view_idx], crop_box))
                        sample['gt_keypoints_2d_rgb'] = np.stack(out, axis=0).astype(np.float32)
            else:
                h, w = frames[0].shape[:2]
                crop_box = self._crop_box_for_aspect(h, w, out_h, out_w)
                x0, y0, crop_w, crop_h = crop_box
                sample[key] = [
                    self._resize_frame(frame[y0:y0 + crop_h, x0:x0 + crop_w], out_w, out_h)
                    for frame in frames
                ]
                if camera_raw is not None:
                    sample[camera_key] = self._transform_camera(camera_raw, x0, y0, crop_w, crop_h, out_w, out_h)
                if self.update_2d_keypoints and key == 'input_rgb' and 'gt_keypoints_2d_rgb' in sample:
                    sample['gt_keypoints_2d_rgb'] = self._transform_points_2d(
                        sample['gt_keypoints_2d_rgb'],
                        (h, w),
                        crop_box,
                    )
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

    def _normalize_frame(self, frame):
        if frame.ndim == 2:
            mean = self.mean[0, 0, 0]
            std = self.std[0, 0, 0]
            return (frame.astype(np.float32) - mean) / std
        return (frame.astype(np.float32) - self.mean) / self.std

    def _normalize_sequence(self, frames):
        normalized_frames = []
        for frame in frames:
            normalized_frames.append(self._normalize_frame(frame))
        return normalized_frames

    def __call__(self, sample):
        for key in self.keys:
            frames = sample[key]
            if (
                isinstance(frames, (list, tuple))
                and len(frames) > 0
                and isinstance(frames[0], (list, tuple))
            ):
                sample[key] = [self._normalize_sequence(view_frames) for view_frames in frames]
            else:
                sample[key] = self._normalize_sequence(frames)
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
    
