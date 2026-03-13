import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


_FRAME_ID_RE = re.compile(r"_(\d{8})$")
_VALID_MODALITIES = {"rgb", "depth", "lidar"}


def _normalize_apply_to(apply_to: Sequence[str]) -> List[str]:
    if not isinstance(apply_to, (list, tuple)):
        raise ValueError(f"`apply_to` must be a sequence of modality names, got {type(apply_to).__name__}.")
    out: List[str] = []
    for item in apply_to:
        name = str(item).strip().lower()
        if name not in _VALID_MODALITIES:
            raise ValueError(f"Unsupported modality in `apply_to`: {item}. Expected one of {_VALID_MODALITIES}.")
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("`apply_to` must contain at least one supported modality.")
    return out


def panoptic_disk_camera_name(camera_name: str) -> str:
    name = str(camera_name).strip().lower()
    if not name.startswith("kinect_"):
        return name
    tail = name.split("_", 1)[1]
    try:
        return f"kinect_{int(tail)}"
    except ValueError:
        return name


def resolve_panoptic_sequence_root(sample: Dict[str, Any], data_root: Optional[str] = None) -> Path:
    if "sequence_root" in sample:
        return Path(sample["sequence_root"]).expanduser().resolve()
    seq_name = sample.get("seq_name")
    if seq_name is None:
        raise ValueError("Sample is missing `seq_name`, required to resolve Panoptic mask paths.")
    if data_root is None:
        raise ValueError(
            "Panoptic mask transform requires either `sequence_root` in sample or `data_root` in transform config."
        )
    return (Path(data_root).expanduser().resolve() / str(seq_name))


def extract_panoptic_frame_ids(sample: Dict[str, Any]) -> List[int]:
    if "body_frame_ids" in sample:
        frame_ids = [int(x) for x in sample["body_frame_ids"]]
        if not frame_ids:
            raise ValueError("`body_frame_ids` is empty.")
        return frame_ids

    sample_id = str(sample.get("sample_id", ""))
    m = _FRAME_ID_RE.search(sample_id)
    if m is None:
        raise ValueError(
            "Could not resolve Panoptic frame ids from sample. Expected `body_frame_ids` or a parseable `sample_id`."
        )
    return [int(m.group(1))]


def resolve_panoptic_mask_path(
    sequence_root: Path,
    camera_name: str,
    frame_id: int,
    mask_subdir: str = "sam_segmentation_mask",
) -> Path:
    return sequence_root / mask_subdir / panoptic_disk_camera_name(camera_name) / f"{int(frame_id):08d}.png"


def load_panoptic_binary_mask(mask_path: Path, expected_hw: Tuple[int, int]) -> np.ndarray:
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing Panoptic segmentation mask: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to decode Panoptic segmentation mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Panoptic segmentation mask must be single-channel, got shape={mask.shape} at {mask_path}")
    expected_h, expected_w = int(expected_hw[0]), int(expected_hw[1])
    if mask.shape != (expected_h, expected_w):
        raise ValueError(
            f"Panoptic segmentation mask shape mismatch for {mask_path}: mask={mask.shape}, frame={(expected_h, expected_w)}"
        )
    return mask > 0


def apply_binary_mask_to_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Frame must be np.ndarray, got {type(frame).__name__}.")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D bool array, got shape={mask.shape}.")
    if frame.shape[:2] != mask.shape:
        raise ValueError(f"Frame/mask shape mismatch: frame={frame.shape}, mask={mask.shape}")
    out = frame.copy()
    if out.ndim == 2:
        out[~mask] = 0
    elif out.ndim == 3:
        out[~mask, ...] = 0
    else:
        raise ValueError(f"Unsupported frame rank for masking: shape={frame.shape}")
    return out


def load_panoptic_camera_meta(sequence_root: Path, camera_name: str) -> Dict[str, np.ndarray | str | int]:
    meta_path = sequence_root / "meta" / "cameras_kinect_cropped.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing Panoptic camera metadata file: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    disk_camera = panoptic_disk_camera_name(camera_name)
    if disk_camera not in meta:
        raise KeyError(f"Camera {disk_camera} not found in Panoptic camera metadata: {meta_path}")
    cam = meta[disk_camera]
    if "K_color" not in cam or "K_depth" not in cam:
        raise KeyError(f"Camera metadata for {disk_camera} is missing K_color/K_depth in {meta_path}")
    return {
        "K_color": np.asarray(cam["K_color"], dtype=np.float32),
        "K_depth": np.asarray(cam["K_depth"], dtype=np.float32),
        "M_color": np.asarray(cam["M_color"], dtype=np.float32),
        "M_depth": np.asarray(cam["M_depth"], dtype=np.float32),
        "color_width": int(cam.get("color_width", 0)),
        "color_height": int(cam.get("color_height", 0)),
        "extrinsic_world_to_color": None if "extrinsic_world_to_color" not in cam else np.asarray(
            cam["extrinsic_world_to_color"], dtype=np.float32
        ),
        "extrinsic_world_to_color_unit": str(cam.get("extrinsic_world_to_color_unit", "cm")).lower(),
    }


def _camera_to_world(points_cam: np.ndarray, extrinsic_world_to_camera: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsic_world_to_camera, dtype=np.float32)
    if ext.shape != (3, 4):
        raise ValueError(f"Expected extrinsic with shape (3,4), got {ext.shape}")
    rot = ext[:, :3]
    trans = ext[:, 3]
    pts = np.asarray(points_cam, dtype=np.float32)
    return (pts - trans[None, :]) @ rot


def _world_to_camera(points_world: np.ndarray, extrinsic_world_to_camera: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsic_world_to_camera, dtype=np.float32)
    if ext.shape != (3, 4):
        raise ValueError(f"Expected extrinsic with shape (3,4), got {ext.shape}")
    rot = ext[:, :3]
    trans = ext[:, 3]
    pts = np.asarray(points_world, dtype=np.float32)
    return (pts @ rot.T) + trans[None, :]


def _scale_color_extrinsic_translation(
    color_extrinsic: np.ndarray,
    extrinsic_unit: str,
    target_unit: str,
) -> np.ndarray:
    ext = np.asarray(color_extrinsic, dtype=np.float32).copy()
    unit_src = str(extrinsic_unit).lower()
    unit_dst = str(target_unit).lower()
    if unit_src == unit_dst:
        return ext
    if unit_src == "cm" and unit_dst == "m":
        ext[:, 3] *= 0.01
        return ext
    if unit_src == "cm" and unit_dst == "mm":
        ext[:, 3] *= 10.0
        return ext
    if unit_src == "m" and unit_dst == "mm":
        ext[:, 3] *= 1000.0
        return ext
    if unit_src == "m" and unit_dst == "cm":
        ext[:, 3] *= 100.0
        return ext
    if unit_src == "mm" and unit_dst == "m":
        ext[:, 3] *= 0.001
        return ext
    if unit_src == "mm" and unit_dst == "cm":
        ext[:, 3] *= 0.1
        return ext
    raise ValueError(f"Unsupported color extrinsic unit conversion: {unit_src} -> {unit_dst}")


def _project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected camera points with shape (N,3), got {pts.shape}")
    k = np.asarray(intrinsic, dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"Expected intrinsic with shape (3,3), got {k.shape}")
    z = pts[:, 2]
    valid = z > 1e-8
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        xy = pts[valid, :2] / z[valid, None]
        uv[valid, 0] = k[0, 0] * xy[:, 0] + k[0, 2]
        uv[valid, 1] = k[1, 1] * xy[:, 1] + k[1, 2]
    return uv, valid


def reproject_rgb_mask_to_depth_mask(
    depth_frame: np.ndarray,
    rgb_mask: np.ndarray,
    k_depth: np.ndarray,
    k_color: np.ndarray,
    depth_extrinsic: np.ndarray,
    color_extrinsic: np.ndarray,
) -> np.ndarray:
    if depth_frame.ndim != 2:
        raise ValueError(f"Depth frame must be single-channel, got {depth_frame.shape}")
    if rgb_mask.ndim != 2:
        raise ValueError(f"RGB mask must be single-channel, got {rgb_mask.shape}")

    h_depth, w_depth = depth_frame.shape
    ys, xs = np.nonzero(depth_frame > 0)
    out = np.zeros((h_depth, w_depth), dtype=bool)
    if ys.size == 0:
        return out

    z = depth_frame[ys, xs].astype(np.float32)
    k_inv = np.linalg.inv(np.asarray(k_depth, dtype=np.float32))
    pixels = np.stack([xs.astype(np.float32), ys.astype(np.float32), np.ones_like(xs, dtype=np.float32)], axis=0)
    rays = k_inv @ pixels
    pts_depth = (rays * z.reshape(1, -1)).T.astype(np.float32)

    pts_world = _camera_to_world(pts_depth, depth_extrinsic)
    pts_color = _world_to_camera(pts_world, color_extrinsic)
    uv, z_valid = _project_points(pts_color, k_color)

    u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
    v = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    in_bounds = (
        z_valid
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (u >= 0)
        & (u < rgb_mask.shape[1])
        & (v >= 0)
        & (v < rgb_mask.shape[0])
    )
    if not np.any(in_bounds):
        return out

    keep = np.zeros(xs.shape[0], dtype=bool)
    keep[in_bounds] = rgb_mask[v[in_bounds], u[in_bounds]]
    out[ys[keep], xs[keep]] = True
    return out


def filter_lidar_points_with_rgb_mask(
    pointcloud: np.ndarray,
    rgb_mask: np.ndarray,
    k_color: np.ndarray,
    lidar_extrinsic: np.ndarray,
    color_extrinsic: np.ndarray,
) -> np.ndarray:
    pc = np.asarray(pointcloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"Point cloud must be (N,C>=3), got {pc.shape}")
    if rgb_mask.ndim != 2:
        raise ValueError(f"RGB mask must be single-channel, got {rgb_mask.shape}")
    if pc.shape[0] == 0:
        return pc.copy()

    pts_lidar = pc[:, :3]
    pts_world = _camera_to_world(pts_lidar, lidar_extrinsic)
    pts_color = _world_to_camera(pts_world, color_extrinsic)
    uv, z_valid = _project_points(pts_color, k_color)

    u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
    v = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    in_bounds = (
        z_valid
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (u >= 0)
        & (u < rgb_mask.shape[1])
        & (v >= 0)
        & (v < rgb_mask.shape[0])
    )
    keep = np.zeros(pc.shape[0], dtype=bool)
    keep[in_bounds] = rgb_mask[v[in_bounds], u[in_bounds]]
    return pc[keep].astype(pointcloud.dtype, copy=False)


def _ensure_view_sequences(frame_data: Any, key_name: str) -> Tuple[List[List[np.ndarray]], bool]:
    if isinstance(frame_data, list):
        if not frame_data:
            raise ValueError(f"`{key_name}` is empty.")
        first = frame_data[0]
        if isinstance(first, np.ndarray):
            return [frame_data], True
        if isinstance(first, list):
            if not all(isinstance(view, list) for view in frame_data):
                raise ValueError(f"`{key_name}` must be list[np.ndarray] or list[list[np.ndarray]].")
            return frame_data, False
        raise ValueError(f"`{key_name}` has unsupported list element type: {type(first).__name__}.")

    raise ValueError(
        f"`{key_name}` must be list[np.ndarray] or list[list[np.ndarray]] before ToTensor, got {type(frame_data).__name__}."
    )


def _restore_view_sequences(view_sequences: List[List[np.ndarray]], single_view: bool) -> Any:
    return view_sequences[0] if single_view else view_sequences


class ApplyPanopticSegmentationMask:
    def __init__(
        self,
        apply_to: Sequence[str],
        data_root: Optional[str] = None,
        mask_subdir: str = "sam_segmentation_mask",
    ):
        self.apply_to = _normalize_apply_to(apply_to)
        self.data_root = None if data_root is None else str(Path(data_root).expanduser().resolve())
        self.mask_subdir = str(mask_subdir).strip().strip("/")
        if not self.mask_subdir:
            raise ValueError("`mask_subdir` must be a non-empty relative directory name.")
        self._camera_meta_cache: Dict[Path, Dict[str, Dict[str, np.ndarray | str | int]]] = {}

    @staticmethod
    def _selected_cameras(sample: Dict[str, Any], modality: str) -> List[str]:
        selected = sample.get("selected_cameras")
        if not isinstance(selected, dict):
            raise ValueError("Sample is missing `selected_cameras`, required for Panoptic mask resolution.")
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
            raise ValueError(f"Sample is missing `{key}`, required for Panoptic depth-mask reprojection.")
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
            return payload
        raise ValueError(f"`{key}` must be dict or list[dict], got {type(payload).__name__}.")

    @staticmethod
    def _camera_payload_map(sample: Dict[str, Any], modality: str) -> Dict[str, Dict[str, Any]]:
        cameras = ApplyPanopticSegmentationMask._selected_cameras(sample, modality)
        payloads = ApplyPanopticSegmentationMask._camera_payloads(sample, modality)
        if len(cameras) != len(payloads):
            raise ValueError(
                f"Camera metadata mismatch for `{modality}`: cameras={len(cameras)}, payloads={len(payloads)}."
            )
        return {cam_name: payload for cam_name, payload in zip(cameras, payloads)}

    def _sequence_camera_meta(self, sequence_root: Path, camera_name: str) -> Dict[str, np.ndarray | str | int]:
        if sequence_root not in self._camera_meta_cache:
            self._camera_meta_cache[sequence_root] = {}
        cache = self._camera_meta_cache[sequence_root]
        disk_camera = panoptic_disk_camera_name(camera_name)
        if disk_camera not in cache:
            cache[disk_camera] = load_panoptic_camera_meta(sequence_root, disk_camera)
        return cache[disk_camera]

    @staticmethod
    def _sample_unit(sample: Dict[str, Any]) -> str:
        unit = str(sample.get("unit", "m")).lower()
        if unit not in {"m", "cm", "mm"}:
            raise ValueError(f"Unsupported Panoptic sample unit for mask reprojection: {unit}")
        return unit

    def _apply_to_modality(
        self,
        sample: Dict[str, Any],
        sample_key: str,
        modality: str,
        frame_ids: List[int],
        sequence_root: Path,
        mask_cache: Dict[Path, np.ndarray],
    ) -> None:
        if sample_key not in sample:
            return

        view_sequences, single_view = _ensure_view_sequences(sample[sample_key], sample_key)
        cameras = self._selected_cameras(sample, modality)
        if len(view_sequences) != len(cameras):
            raise ValueError(
                f"View/camera count mismatch for `{sample_key}`: views={len(view_sequences)}, cameras={len(cameras)}."
            )

        masked_views: List[List[np.ndarray]] = []
        for camera_name, frames in zip(cameras, view_sequences):
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
                mask_path = resolve_panoptic_mask_path(
                    sequence_root=sequence_root,
                    camera_name=camera_name,
                    frame_id=frame_id,
                    mask_subdir=self.mask_subdir,
                )
                if mask_path not in mask_cache:
                    mask_cache[mask_path] = load_panoptic_binary_mask(mask_path, frame.shape[:2])
                masked_frames.append(apply_binary_mask_to_frame(frame, mask_cache[mask_path]))
            masked_views.append(masked_frames)
        sample[sample_key] = _restore_view_sequences(masked_views, single_view)

    def _apply_to_depth(
        self,
        sample: Dict[str, Any],
        frame_ids: List[int],
        sequence_root: Path,
        mask_cache: Dict[Path, np.ndarray],
    ) -> None:
        sample_key = "input_depth"
        if sample_key not in sample:
            return

        view_sequences, single_view = _ensure_view_sequences(sample[sample_key], sample_key)
        cameras = self._selected_cameras(sample, "depth")
        camera_payloads = self._camera_payloads(sample, "depth")
        if len(view_sequences) != len(cameras) or len(camera_payloads) != len(cameras):
            raise ValueError(
                f"Depth view/camera metadata mismatch: views={len(view_sequences)}, cameras={len(cameras)}, "
                f"camera_payloads={len(camera_payloads)}."
            )

        sample_unit = self._sample_unit(sample)
        ref_rgb_names, ref_rgb_payloads = self._resolve_reference_rgb_cameras(sample, "depth")
        masked_views: List[List[np.ndarray]] = []
        for camera_name, frames, depth_camera, rgb_name, rgb_camera in zip(
            cameras, view_sequences, camera_payloads, ref_rgb_names, ref_rgb_payloads
        ):
            depth_meta = self._sequence_camera_meta(sequence_root, camera_name)
            rgb_meta = self._sequence_camera_meta(sequence_root, rgb_name)
            color_ext = np.asarray(rgb_camera.get("extrinsic"), dtype=np.float32)
            if color_ext.shape != (3, 4):
                fallback_color_ext = rgb_meta.get("extrinsic_world_to_color")
                if fallback_color_ext is None:
                    raise KeyError(
                        f"Camera metadata for {sequence_root.name}/{rgb_name} is missing `extrinsic_world_to_color`."
                    )
                color_ext = _scale_color_extrinsic_translation(
                    np.asarray(fallback_color_ext, dtype=np.float32),
                    str(rgb_meta["extrinsic_world_to_color_unit"]),
                    sample_unit,
                )
            k_color = np.asarray(rgb_meta["K_color"], dtype=np.float32)
            k_depth = np.asarray(depth_meta["K_depth"], dtype=np.float32)
            color_h = int(rgb_meta.get("color_height", 0))
            color_w = int(rgb_meta.get("color_width", 0))
            if color_h <= 0 or color_w <= 0:
                raise ValueError(
                    f"Camera metadata for {sequence_root.name}/{rgb_name} is missing valid color_width/color_height."
                )
            depth_ext = np.asarray(depth_camera.get("extrinsic"), dtype=np.float32)
            if depth_ext.shape != (3, 4):
                raise ValueError(
                    f"Depth camera extrinsic for {sequence_root.name}/{camera_name} must be (3,4), got {depth_ext.shape}."
                )

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
                if frame.ndim != 2:
                    raise ValueError(
                        f"`{sample_key}` camera={camera_name} frame={frame_id} must be single-channel, got {frame.shape}."
                    )
                mask_path = resolve_panoptic_mask_path(
                    sequence_root=sequence_root,
                    camera_name=rgb_name,
                    frame_id=frame_id,
                    mask_subdir=self.mask_subdir,
                )
                if mask_path not in mask_cache:
                    mask_cache[mask_path] = load_panoptic_binary_mask(mask_path, (color_h, color_w))
                rgb_mask = mask_cache[mask_path]
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

    @staticmethod
    def _resolve_reference_rgb_cameras(
        sample: Dict[str, Any],
        target_modality: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        rgb_names = ApplyPanopticSegmentationMask._selected_cameras(sample, "rgb")
        rgb_payloads = ApplyPanopticSegmentationMask._camera_payloads(sample, "rgb")
        target_names = ApplyPanopticSegmentationMask._selected_cameras(sample, target_modality)
        if len(rgb_names) == 1:
            return [rgb_names[0]] * len(target_names), [rgb_payloads[0]] * len(target_names)
        if len(rgb_names) == len(target_names):
            return list(rgb_names), list(rgb_payloads)

        rgb_map = {name: payload for name, payload in zip(rgb_names, rgb_payloads)}
        ref_names: List[str] = []
        ref_payloads: List[Dict[str, Any]] = []
        for target_name in target_names:
            if target_name not in rgb_map:
                raise ValueError(
                    f"Cannot resolve reference RGB camera for {target_modality} camera `{target_name}`. "
                    f"Available RGB cameras: {rgb_names}."
                )
            ref_names.append(target_name)
            ref_payloads.append(rgb_map[target_name])
        return ref_names, ref_payloads

    def _apply_to_lidar(
        self,
        sample: Dict[str, Any],
        frame_ids: List[int],
        sequence_root: Path,
        mask_cache: Dict[Path, np.ndarray],
    ) -> None:
        sample_key = "input_lidar"
        if sample_key not in sample:
            return

        view_sequences, single_view = _ensure_view_sequences(sample[sample_key], sample_key)
        lidar_cameras = self._selected_cameras(sample, "lidar")
        lidar_payloads = self._camera_payloads(sample, "lidar")
        if len(view_sequences) != len(lidar_cameras) or len(lidar_payloads) != len(lidar_cameras):
            raise ValueError(
                f"LiDAR view/camera metadata mismatch: views={len(view_sequences)}, cameras={len(lidar_cameras)}, "
                f"camera_payloads={len(lidar_payloads)}."
            )

        sample_unit = self._sample_unit(sample)
        ref_rgb_names, ref_rgb_payloads = self._resolve_reference_rgb_cameras(sample, "lidar")
        masked_views: List[List[np.ndarray]] = []
        for lidar_name, frames, lidar_camera, rgb_name, rgb_camera in zip(
            lidar_cameras, view_sequences, lidar_payloads, ref_rgb_names, ref_rgb_payloads
        ):
            rgb_meta = self._sequence_camera_meta(sequence_root, rgb_name)
            color_ext = np.asarray(rgb_camera.get("extrinsic"), dtype=np.float32)
            if color_ext.shape != (3, 4):
                fallback_color_ext = rgb_meta.get("extrinsic_world_to_color")
                if fallback_color_ext is None:
                    raise ValueError(
                        f"RGB camera extrinsic for {sequence_root.name}/{rgb_name} must be (3,4), got {color_ext.shape}."
                    )
                color_ext = _scale_color_extrinsic_translation(
                    np.asarray(fallback_color_ext, dtype=np.float32),
                    str(rgb_meta["extrinsic_world_to_color_unit"]),
                    sample_unit,
                )
            k_color = np.asarray(rgb_meta["K_color"], dtype=np.float32)
            color_h = int(rgb_meta.get("color_height", 0))
            color_w = int(rgb_meta.get("color_width", 0))
            if color_h <= 0 or color_w <= 0:
                raise ValueError(
                    f"Camera metadata for {sequence_root.name}/{rgb_name} is missing valid color_width/color_height."
                )
            lidar_ext = np.asarray(lidar_camera.get("extrinsic"), dtype=np.float32)
            if lidar_ext.shape != (3, 4):
                raise ValueError(
                    f"LiDAR camera extrinsic for {sequence_root.name}/{lidar_name} must be (3,4), got {lidar_ext.shape}."
                )
            if len(frames) != len(frame_ids):
                raise ValueError(
                    f"Frame count mismatch for `{sample_key}` camera={lidar_name}: frames={len(frames)}, "
                    f"frame_ids={len(frame_ids)}."
                )

            masked_frames: List[np.ndarray] = []
            for frame_id, frame in zip(frame_ids, frames):
                if not isinstance(frame, np.ndarray):
                    raise ValueError(
                        f"`{sample_key}` camera={lidar_name} frame={frame_id} must be np.ndarray, got {type(frame).__name__}."
                    )
                mask_path = resolve_panoptic_mask_path(
                    sequence_root=sequence_root,
                    camera_name=rgb_name,
                    frame_id=frame_id,
                    mask_subdir=self.mask_subdir,
                )
                if mask_path not in mask_cache:
                    mask_cache[mask_path] = load_panoptic_binary_mask(mask_path, (color_h, color_w))
                rgb_mask = mask_cache[mask_path]
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
        sequence_root = resolve_panoptic_sequence_root(sample, data_root=self.data_root)
        frame_ids = extract_panoptic_frame_ids(sample)
        mask_cache: Dict[Path, np.ndarray] = {}

        if "rgb" in self.apply_to:
            self._apply_to_modality(
                sample=sample,
                sample_key="input_rgb",
                modality="rgb",
                frame_ids=frame_ids,
                sequence_root=sequence_root,
                mask_cache=mask_cache,
            )
        if "depth" in self.apply_to:
            self._apply_to_depth(sample, frame_ids, sequence_root, mask_cache)
        if "lidar" in self.apply_to:
            self._apply_to_lidar(sample, frame_ids, sequence_root, mask_cache)
        return sample
