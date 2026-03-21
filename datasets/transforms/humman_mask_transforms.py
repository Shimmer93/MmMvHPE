from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .panoptic_mask_transforms import (
    _ensure_view_sequences,
    _normalize_apply_to,
    _restore_view_sequences,
    apply_binary_mask_to_frame,
    filter_lidar_points_with_rgb_mask,
    reproject_rgb_mask_to_depth_mask,
)


def resolve_humman_mask_root(
    sample: Dict[str, Any],
    data_root: Optional[str] = None,
    mask_root: Optional[str] = None,
    mask_subdir: str = "sam_segmentation_mask",
) -> Path:
    if mask_root is not None:
        return Path(mask_root).expanduser().resolve()
    if "mask_root" in sample:
        return Path(sample["mask_root"]).expanduser().resolve()
    if data_root is None:
        raise ValueError(
            "HuMMan mask transform requires either `mask_root` in the transform config/sample or `data_root`."
        )
    return (Path(data_root).expanduser().resolve() / mask_subdir)


def resolve_humman_mask_path(mask_root: Path, seq_name: str, camera_name: str, frame_id: int) -> Path:
    if not seq_name:
        raise ValueError("HuMMan mask resolution requires a non-empty `seq_name`.")
    return mask_root / f"{seq_name}_{camera_name}_{int(frame_id):06d}.png"


def load_humman_binary_mask(mask_path: Path, expected_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing HuMMan segmentation mask: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to decode HuMMan segmentation mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"HuMMan segmentation mask must be single-channel, got shape={mask.shape} at {mask_path}")
    if expected_hw is not None:
        expected_h, expected_w = int(expected_hw[0]), int(expected_hw[1])
        if mask.shape != (expected_h, expected_w):
            raise ValueError(
                f"HuMMan segmentation mask shape mismatch for {mask_path}: "
                f"mask={mask.shape}, frame={(expected_h, expected_w)}"
            )
    return mask > 0


def _get_or_load_mask(
    mask_cache: Dict[Path, np.ndarray],
    mask_path: Path,
    expected_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    if mask_path not in mask_cache:
        mask_cache[mask_path] = load_humman_binary_mask(mask_path, expected_hw=expected_hw)
    elif expected_hw is not None and mask_cache[mask_path].shape != tuple(int(x) for x in expected_hw):
        raise ValueError(
            f"HuMMan segmentation mask shape mismatch for {mask_path}: "
            f"mask={mask_cache[mask_path].shape}, frame={tuple(int(x) for x in expected_hw)}"
        )
    return mask_cache[mask_path]


class ApplyHummanSegmentationMask:
    def __init__(
        self,
        apply_to: Sequence[str],
        data_root: Optional[str] = None,
        mask_root: Optional[str] = None,
        mask_subdir: str = "sam_segmentation_mask",
    ):
        self.apply_to = _normalize_apply_to(apply_to)
        self.data_root = None if data_root is None else str(Path(data_root).expanduser().resolve())
        self.mask_root = None if mask_root is None else str(Path(mask_root).expanduser().resolve())
        self.mask_subdir = str(mask_subdir).strip().strip("/")
        if not self.mask_subdir:
            raise ValueError("`mask_subdir` must be a non-empty relative directory name.")

    @staticmethod
    def _selected_cameras(sample: Dict[str, Any], modality: str) -> List[str]:
        selected = sample.get("selected_cameras")
        if not isinstance(selected, dict):
            raise ValueError("Sample is missing `selected_cameras`, required for HuMMan mask resolution.")
        cameras = selected.get(modality)
        if cameras is None:
            raise ValueError(f"Sample `selected_cameras` has no `{modality}` entry.")
        if not isinstance(cameras, list) or not all(isinstance(c, str) for c in cameras):
            raise ValueError(f"`selected_cameras[{modality!r}]` must be list[str], got {cameras!r}.")
        return cameras

    @staticmethod
    def _camera_payloads(sample: Dict[str, Any], modality: str) -> List[Dict[str, Any]]:
        key = f"{modality}_camera"
        payload = sample.get(key)
        if payload is None:
            raise ValueError(f"Sample is missing `{key}`, required for HuMMan mask reprojection.")
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
            return payload
        raise ValueError(f"`{key}` must be dict or list[dict], got {type(payload).__name__}.")

    @staticmethod
    def _frame_ids_for_cameras(sample: Dict[str, Any], modality: str) -> List[List[int]]:
        selected_frame_ids = sample.get("selected_frame_ids")
        if not isinstance(selected_frame_ids, dict):
            raise ValueError("Sample is missing `selected_frame_ids`, required for HuMMan mask resolution.")
        frame_map = selected_frame_ids.get(modality)
        if not isinstance(frame_map, dict):
            raise ValueError(f"Sample `selected_frame_ids` has no `{modality}` mapping.")

        frame_ids: List[List[int]] = []
        for camera_name in ApplyHummanSegmentationMask._selected_cameras(sample, modality):
            camera_frame_ids = frame_map.get(camera_name)
            if not isinstance(camera_frame_ids, list) or not camera_frame_ids:
                raise ValueError(
                    f"Sample `selected_frame_ids[{modality!r}]` is missing a non-empty list for camera `{camera_name}`."
                )
            frame_ids.append([int(frame_id) for frame_id in camera_frame_ids])
        return frame_ids

    @staticmethod
    def _resolve_reference_rgb_cameras(
        sample: Dict[str, Any],
        target_modality: str,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[List[int]]]:
        rgb_names = ApplyHummanSegmentationMask._selected_cameras(sample, "rgb")
        rgb_payloads = ApplyHummanSegmentationMask._camera_payloads(sample, "rgb")
        rgb_frame_ids = ApplyHummanSegmentationMask._frame_ids_for_cameras(sample, "rgb")
        target_names = ApplyHummanSegmentationMask._selected_cameras(sample, target_modality)

        if len(rgb_names) != len(rgb_payloads) or len(rgb_names) != len(rgb_frame_ids):
            raise ValueError(
                f"RGB camera metadata mismatch: cameras={len(rgb_names)}, payloads={len(rgb_payloads)}, "
                f"frame_id_lists={len(rgb_frame_ids)}."
            )

        if len(rgb_names) == 1:
            return [rgb_names[0]] * len(target_names), [rgb_payloads[0]] * len(target_names), [rgb_frame_ids[0]] * len(target_names)

        rgb_map = {
            name: (payload, frame_ids)
            for name, payload, frame_ids in zip(rgb_names, rgb_payloads, rgb_frame_ids)
        }
        ref_names: List[str] = []
        ref_payloads: List[Dict[str, Any]] = []
        ref_frame_ids: List[List[int]] = []
        for target_name in target_names:
            if target_name not in rgb_map:
                raise ValueError(
                    f"Cannot resolve reference RGB camera for {target_modality} camera `{target_name}`. "
                    f"Available RGB cameras: {rgb_names}."
                )
            payload, frame_ids = rgb_map[target_name]
            ref_names.append(target_name)
            ref_payloads.append(payload)
            ref_frame_ids.append(frame_ids)
        return ref_names, ref_payloads, ref_frame_ids

    @staticmethod
    def _sample_unit(sample: Dict[str, Any]) -> str:
        unit = str(sample.get("unit", "m")).lower()
        if unit != "m":
            raise ValueError(
                f"ApplyHummanSegmentationMask currently supports only sample unit `m`, got `{unit}`."
            )
        return unit

    @staticmethod
    def _rgb_frame_shape_map(sample: Dict[str, Any]) -> Dict[str, List[Tuple[int, int]]]:
        if "input_rgb" not in sample:
            return {}
        view_sequences, _ = _ensure_view_sequences(sample["input_rgb"], "input_rgb")
        cameras = ApplyHummanSegmentationMask._selected_cameras(sample, "rgb")
        if len(view_sequences) != len(cameras):
            raise ValueError(
                f"RGB view/camera count mismatch: views={len(view_sequences)}, cameras={len(cameras)}."
            )
        out: Dict[str, List[Tuple[int, int]]] = {}
        for camera_name, frames in zip(cameras, view_sequences):
            shapes: List[Tuple[int, int]] = []
            for frame in frames:
                if not isinstance(frame, np.ndarray):
                    raise ValueError(f"`input_rgb` camera={camera_name} contains non-array frame {type(frame).__name__}.")
                shapes.append((int(frame.shape[0]), int(frame.shape[1])))
            out[camera_name] = shapes
        return out

    def _apply_to_rgb(
        self,
        sample: Dict[str, Any],
        seq_name: str,
        mask_root: Path,
        mask_cache: Dict[Path, np.ndarray],
    ) -> None:
        sample_key = "input_rgb"
        if sample_key not in sample:
            return

        view_sequences, single_view = _ensure_view_sequences(sample[sample_key], sample_key)
        cameras = self._selected_cameras(sample, "rgb")
        frame_id_lists = self._frame_ids_for_cameras(sample, "rgb")
        if len(view_sequences) != len(cameras) or len(frame_id_lists) != len(cameras):
            raise ValueError(
                f"RGB view/camera metadata mismatch: views={len(view_sequences)}, cameras={len(cameras)}, "
                f"frame_id_lists={len(frame_id_lists)}."
            )

        masked_views: List[List[np.ndarray]] = []
        for camera_name, frames, frame_ids in zip(cameras, view_sequences, frame_id_lists):
            if len(frames) != len(frame_ids):
                raise ValueError(
                    f"Frame count mismatch for `{sample_key}` camera={camera_name}: frames={len(frames)}, "
                    f"frame_ids={len(frame_ids)}."
                )
            masked_frames: List[np.ndarray] = []
            for frame_id, frame in zip(frame_ids, frames):
                if not isinstance(frame, np.ndarray):
                    raise ValueError(
                        f"`{sample_key}` camera={camera_name} frame={frame_id} must be np.ndarray, got {type(frame).__name__}."
                    )
                mask_path = resolve_humman_mask_path(mask_root, seq_name, camera_name, frame_id)
                mask = _get_or_load_mask(mask_cache, mask_path, expected_hw=frame.shape[:2])
                masked_frames.append(apply_binary_mask_to_frame(frame, mask))
            masked_views.append(masked_frames)
        sample[sample_key] = _restore_view_sequences(masked_views, single_view)

    def _apply_to_depth(
        self,
        sample: Dict[str, Any],
        seq_name: str,
        mask_root: Path,
        mask_cache: Dict[Path, np.ndarray],
    ) -> None:
        sample_key = "input_depth"
        if sample_key not in sample:
            return

        self._sample_unit(sample)
        view_sequences, single_view = _ensure_view_sequences(sample[sample_key], sample_key)
        cameras = self._selected_cameras(sample, "depth")
        camera_payloads = self._camera_payloads(sample, "depth")
        frame_id_lists = self._frame_ids_for_cameras(sample, "depth")
        if len(view_sequences) != len(cameras) or len(camera_payloads) != len(cameras) or len(frame_id_lists) != len(cameras):
            raise ValueError(
                f"Depth view/camera metadata mismatch: views={len(view_sequences)}, cameras={len(cameras)}, "
                f"camera_payloads={len(camera_payloads)}, frame_id_lists={len(frame_id_lists)}."
            )

        ref_rgb_names, ref_rgb_payloads, ref_rgb_frame_ids = self._resolve_reference_rgb_cameras(sample, "depth")
        rgb_frame_shapes = self._rgb_frame_shape_map(sample)
        masked_views: List[List[np.ndarray]] = []
        for camera_name, frames, depth_camera, depth_frame_ids, rgb_name, rgb_camera, rgb_frame_ids in zip(
            cameras, view_sequences, camera_payloads, frame_id_lists, ref_rgb_names, ref_rgb_payloads, ref_rgb_frame_ids
        ):
            if len(frames) != len(depth_frame_ids):
                raise ValueError(
                    f"Frame count mismatch for `{sample_key}` camera={camera_name}: frames={len(frames)}, "
                    f"frame_ids={len(depth_frame_ids)}."
                )
            if len(depth_frame_ids) != len(rgb_frame_ids):
                raise ValueError(
                    f"RGB/depth frame count mismatch for camera pair `{rgb_name}` -> `{camera_name}`: "
                    f"rgb_frame_ids={len(rgb_frame_ids)}, depth_frame_ids={len(depth_frame_ids)}."
                )

            k_depth = np.asarray(depth_camera.get("intrinsic"), dtype=np.float32)
            k_color = np.asarray(rgb_camera.get("intrinsic"), dtype=np.float32)
            depth_ext = np.asarray(depth_camera.get("extrinsic"), dtype=np.float32)
            color_ext = np.asarray(rgb_camera.get("extrinsic"), dtype=np.float32)
            if k_depth.shape != (3, 3):
                raise ValueError(f"Depth camera intrinsic for `{camera_name}` must be (3,3), got {k_depth.shape}.")
            if k_color.shape != (3, 3):
                raise ValueError(f"RGB camera intrinsic for `{rgb_name}` must be (3,3), got {k_color.shape}.")
            if depth_ext.shape != (3, 4):
                raise ValueError(f"Depth camera extrinsic for `{camera_name}` must be (3,4), got {depth_ext.shape}.")
            if color_ext.shape != (3, 4):
                raise ValueError(f"RGB camera extrinsic for `{rgb_name}` must be (3,4), got {color_ext.shape}.")

            expected_rgb_shapes = rgb_frame_shapes.get(rgb_name)
            if expected_rgb_shapes is not None and len(expected_rgb_shapes) != len(rgb_frame_ids):
                raise ValueError(
                    f"RGB frame shape metadata mismatch for `{rgb_name}`: shapes={len(expected_rgb_shapes)}, "
                    f"frame_ids={len(rgb_frame_ids)}."
                )

            masked_frames: List[np.ndarray] = []
            for idx, (depth_frame_id, rgb_frame_id, frame) in enumerate(zip(depth_frame_ids, rgb_frame_ids, frames)):
                if not isinstance(frame, np.ndarray):
                    raise ValueError(
                        f"`{sample_key}` camera={camera_name} frame={depth_frame_id} must be np.ndarray, got {type(frame).__name__}."
                    )
                if frame.ndim != 2:
                    raise ValueError(
                        f"`{sample_key}` camera={camera_name} frame={depth_frame_id} must be single-channel, got {frame.shape}."
                    )
                expected_rgb_hw = None if expected_rgb_shapes is None else expected_rgb_shapes[idx]
                mask_path = resolve_humman_mask_path(mask_root, seq_name, rgb_name, rgb_frame_id)
                rgb_mask = _get_or_load_mask(mask_cache, mask_path, expected_hw=expected_rgb_hw)
                depth_mask = reproject_rgb_mask_to_depth_mask(
                    depth_frame=frame,
                    rgb_mask=rgb_mask,
                    k_depth=k_depth,
                    k_color=k_color,
                    depth_extrinsic=depth_ext,
                    color_extrinsic=color_ext,
                )
                masked_frames.append(apply_binary_mask_to_frame(frame, depth_mask))
            masked_views.append(masked_frames)
        sample[sample_key] = _restore_view_sequences(masked_views, single_view)

    def _apply_to_lidar(
        self,
        sample: Dict[str, Any],
        seq_name: str,
        mask_root: Path,
        mask_cache: Dict[Path, np.ndarray],
    ) -> None:
        sample_key = "input_lidar"
        if sample_key not in sample:
            return

        self._sample_unit(sample)
        view_sequences, single_view = _ensure_view_sequences(sample[sample_key], sample_key)
        lidar_cameras = self._selected_cameras(sample, "lidar")
        lidar_payloads = self._camera_payloads(sample, "lidar")
        lidar_frame_ids = self._frame_ids_for_cameras(sample, "lidar")
        if len(view_sequences) != len(lidar_cameras) or len(lidar_payloads) != len(lidar_cameras) or len(lidar_frame_ids) != len(lidar_cameras):
            raise ValueError(
                f"LiDAR view/camera metadata mismatch: views={len(view_sequences)}, cameras={len(lidar_cameras)}, "
                f"camera_payloads={len(lidar_payloads)}, frame_id_lists={len(lidar_frame_ids)}."
            )

        ref_rgb_names, ref_rgb_payloads, ref_rgb_frame_ids = self._resolve_reference_rgb_cameras(sample, "lidar")
        rgb_frame_shapes = self._rgb_frame_shape_map(sample)
        masked_views: List[List[np.ndarray]] = []
        for lidar_name, frames, lidar_camera, lidar_ids, rgb_name, rgb_camera, rgb_ids in zip(
            lidar_cameras, view_sequences, lidar_payloads, lidar_frame_ids, ref_rgb_names, ref_rgb_payloads, ref_rgb_frame_ids
        ):
            if len(frames) != len(lidar_ids):
                raise ValueError(
                    f"Frame count mismatch for `{sample_key}` camera={lidar_name}: frames={len(frames)}, "
                    f"frame_ids={len(lidar_ids)}."
                )
            if len(lidar_ids) != len(rgb_ids):
                raise ValueError(
                    f"RGB/LiDAR frame count mismatch for camera pair `{rgb_name}` -> `{lidar_name}`: "
                    f"rgb_frame_ids={len(rgb_ids)}, lidar_frame_ids={len(lidar_ids)}."
                )

            k_color = np.asarray(rgb_camera.get("intrinsic"), dtype=np.float32)
            color_ext = np.asarray(rgb_camera.get("extrinsic"), dtype=np.float32)
            lidar_ext = np.asarray(lidar_camera.get("extrinsic"), dtype=np.float32)
            if k_color.shape != (3, 3):
                raise ValueError(f"RGB camera intrinsic for `{rgb_name}` must be (3,3), got {k_color.shape}.")
            if color_ext.shape != (3, 4):
                raise ValueError(f"RGB camera extrinsic for `{rgb_name}` must be (3,4), got {color_ext.shape}.")
            if lidar_ext.shape != (3, 4):
                raise ValueError(f"LiDAR camera extrinsic for `{lidar_name}` must be (3,4), got {lidar_ext.shape}.")

            expected_rgb_shapes = rgb_frame_shapes.get(rgb_name)
            if expected_rgb_shapes is not None and len(expected_rgb_shapes) != len(rgb_ids):
                raise ValueError(
                    f"RGB frame shape metadata mismatch for `{rgb_name}`: shapes={len(expected_rgb_shapes)}, "
                    f"frame_ids={len(rgb_ids)}."
                )

            masked_frames: List[np.ndarray] = []
            for idx, (lidar_frame_id, rgb_frame_id, frame) in enumerate(zip(lidar_ids, rgb_ids, frames)):
                if not isinstance(frame, np.ndarray):
                    raise ValueError(
                        f"`{sample_key}` camera={lidar_name} frame={lidar_frame_id} must be np.ndarray, got {type(frame).__name__}."
                    )
                expected_rgb_hw = None if expected_rgb_shapes is None else expected_rgb_shapes[idx]
                mask_path = resolve_humman_mask_path(mask_root, seq_name, rgb_name, rgb_frame_id)
                rgb_mask = _get_or_load_mask(mask_cache, mask_path, expected_hw=expected_rgb_hw)
                masked_frames.append(
                    filter_lidar_points_with_rgb_mask(
                        pointcloud=frame,
                        rgb_mask=rgb_mask,
                        k_color=k_color,
                        lidar_extrinsic=lidar_ext,
                        color_extrinsic=color_ext,
                    )
                )
            masked_views.append(masked_frames)
        sample[sample_key] = _restore_view_sequences(masked_views, single_view)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        seq_name = str(sample.get("seq_name", "")).strip()
        mask_root = resolve_humman_mask_root(
            sample,
            data_root=self.data_root,
            mask_root=self.mask_root,
            mask_subdir=self.mask_subdir,
        )
        mask_cache: Dict[Path, np.ndarray] = {}

        if "rgb" in self.apply_to:
            self._apply_to_rgb(sample, seq_name, mask_root, mask_cache)
        if "depth" in self.apply_to:
            self._apply_to_depth(sample, seq_name, mask_root, mask_cache)
        if "lidar" in self.apply_to:
            self._apply_to_lidar(sample, seq_name, mask_root, mask_cache)
        return sample
