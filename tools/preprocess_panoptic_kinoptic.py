#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.image_detection import YOLO, predict_human_bboxes


SEQUENCE_RE = re.compile(r"^[0-9]{6}_[A-Za-z0-9_]+$")


@dataclass
class SequencePaths:
    seq_name: str
    seq_dir: Path
    sync_path: Path
    calib_path: Path
    panoptic_calib_path: Path
    body_dir: Path
    video_paths: dict[str, Path]
    depth_paths: dict[str, Path]


@dataclass
class BodyFrame:
    body_frame_id: int
    univ_time: float
    joints19: list[float]


@dataclass
class SyncEntry:
    body_frame_id: int
    body_univ_time: float
    color_frame_idx: int
    color_univ_time: float
    depth_frame_idx: int
    depth_univ_time: float
    color_delta_ms: float
    depth_delta_ms: float


def _adjust_camera_intrinsic_for_crop(
    k: np.ndarray, crop_x0: int, crop_y0: int, crop_size: int, out_size: tuple[int, int]
) -> np.ndarray:
    out_w, out_h = out_size
    scale_x = float(out_w) / float(crop_size)
    scale_y = float(out_h) / float(crop_size)
    k_new = k.copy().astype(np.float64)
    k_new[0, 0] *= scale_x
    k_new[1, 1] *= scale_y
    k_new[0, 2] = (k_new[0, 2] - crop_x0) * scale_x
    k_new[1, 2] = (k_new[1, 2] - crop_y0) * scale_y
    k_new[0, 1] *= scale_x
    k_new[1, 0] *= scale_y
    return k_new


def _compute_square_crop(bbox: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    size = int(np.ceil(max(w, h)))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    x0 = int(np.floor(cx - size * 0.5))
    y0 = int(np.floor(cy - size * 0.5))
    x1i = x0 + size
    y1i = y0 + size
    return x0, y0, x1i, y1i, size


def _crop_with_padding(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad_value: int | tuple[int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    x0_clamped = max(0, x0)
    y0_clamped = max(0, y0)
    x1_clamped = min(w, x1)
    y1_clamped = min(h, y1)
    cropped = image[y0_clamped:y1_clamped, x0_clamped:x1_clamped]
    if pad_left or pad_top or pad_right or pad_bottom:
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )
    return cropped


def _parse_sequence_names(root_dir: Path, sequences_csv: str | None, sequence_list: Path | None, max_sequences: int | None) -> list[str]:
    names: set[str] = set()
    if sequences_csv:
        for item in sequences_csv.split(","):
            seq = item.strip()
            if seq:
                names.add(seq)
    if sequence_list:
        if not sequence_list.is_file():
            raise FileNotFoundError(f"sequence list not found: {sequence_list}")
        for raw in sequence_list.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                names.add(line)
    if not names:
        names = {p.name for p in root_dir.iterdir() if p.is_dir() and SEQUENCE_RE.match(p.name)}
    seqs = sorted(names)
    invalid = [s for s in seqs if SEQUENCE_RE.match(s) is None]
    if invalid:
        raise ValueError(f"invalid sequence names: {invalid[:10]}")
    if max_sequences is not None:
        seqs = seqs[:max_sequences]
    return seqs


def _discover_sequence_paths(root_dir: Path, seq_name: str) -> SequencePaths:
    seq_dir = root_dir / seq_name
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"sequence directory not found: {seq_dir}")
    sync_path = seq_dir / f"ksynctables_{seq_name}.json"
    calib_path = seq_dir / f"kcalibration_{seq_name}.json"
    panoptic_calib_path = seq_dir / f"calibration_{seq_name}.json"
    body_dir = seq_dir / "hdPose3d_stage1_coco19"
    if not sync_path.is_file():
        raise FileNotFoundError(f"missing sync table: {sync_path}")
    if not calib_path.is_file():
        raise FileNotFoundError(f"missing calibration: {calib_path}")
    if not panoptic_calib_path.is_file():
        raise FileNotFoundError(f"missing panoptic calibration: {panoptic_calib_path}")
    if not body_dir.is_dir():
        raise FileNotFoundError(f"missing body annotation directory: {body_dir}")
    body_files = list(body_dir.rglob("body3DScene_*.json"))
    if not body_files:
        raise FileNotFoundError(f"no body3DScene files under {body_dir}")

    video_paths: dict[str, Path] = {}
    depth_paths: dict[str, Path] = {}
    for node in range(1, 11):
        node_name = f"KINECTNODE{node}"
        video_path = seq_dir / "kinectVideos" / f"kinect_50_{node:02d}.mp4"
        depth_path = seq_dir / "kinect_shared_depth" / node_name / "depthdata.dat"
        if video_path.is_file():
            video_paths[node_name] = video_path
        if depth_path.is_file():
            depth_paths[node_name] = depth_path

    paired_nodes = sorted(set(video_paths.keys()) & set(depth_paths.keys()))
    if not paired_nodes:
        raise FileNotFoundError(
            f"no Kinect node with both RGB video and depthdata.dat under {seq_dir}"
        )
    video_paths = {k: video_paths[k] for k in paired_nodes}
    depth_paths = {k: depth_paths[k] for k in paired_nodes}

    return SequencePaths(
        seq_name=seq_name,
        seq_dir=seq_dir,
        sync_path=sync_path,
        calib_path=calib_path,
        panoptic_calib_path=panoptic_calib_path,
        body_dir=body_dir,
        video_paths=video_paths,
        depth_paths=depth_paths,
    )


def _load_body_frames(body_dir: Path, max_body_frames: int | None) -> list[BodyFrame]:
    files = sorted(body_dir.rglob("body3DScene_*.json"))
    if max_body_frames is not None:
        files = files[:max_body_frames]
    out: list[BodyFrame] = []
    for file_path in files:
        frame_id = int(file_path.stem.split("_")[-1])
        with file_path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        bodies = d.get("bodies")
        if not isinstance(bodies, list):
            raise ValueError(f"`bodies` must be list in {file_path}")
        if len(bodies) == 0:
            continue
        if len(bodies) > 1:
            raise ValueError(f"sequence is not single-actor at {file_path}: bodies={len(bodies)}")
        joints19 = bodies[0].get("joints19")
        if not isinstance(joints19, list) or len(joints19) != 76:
            raise ValueError(f"invalid joints19 in {file_path}")
        univ_time = d.get("univTime")
        if not isinstance(univ_time, (int, float)):
            raise ValueError(f"invalid univTime in {file_path}")
        out.append(BodyFrame(body_frame_id=frame_id, univ_time=float(univ_time), joints19=joints19))
    if not out:
        raise ValueError(f"no valid single-actor body frames in {body_dir}")
    return out


def _nearest_index(univ_times: np.ndarray, target: float) -> tuple[int, float]:
    idx = int(np.argmin(np.abs(univ_times - target)))
    return idx, float(univ_times[idx])


def _build_sync_entries_for_node(
    body_frames: list[BodyFrame],
    color_univ_times: np.ndarray,
    depth_univ_times: np.ndarray,
    max_sync_delta_ms: float,
) -> list[SyncEntry]:
    entries: list[SyncEntry] = []
    for bf in body_frames:
        if bf.univ_time < 0:
            continue
        cidx, ctime = _nearest_index(color_univ_times, bf.univ_time)
        didx, dtime = _nearest_index(depth_univ_times, bf.univ_time)
        cdelta = abs(ctime - bf.univ_time)
        ddelta = abs(dtime - bf.univ_time)
        if ctime < 0 or dtime < 0:
            continue
        if cdelta > max_sync_delta_ms or ddelta > max_sync_delta_ms:
            continue
        entries.append(
            SyncEntry(
                body_frame_id=bf.body_frame_id,
                body_univ_time=bf.univ_time,
                color_frame_idx=cidx,
                color_univ_time=ctime,
                depth_frame_idx=didx,
                depth_univ_time=dtime,
                color_delta_ms=cdelta,
                depth_delta_ms=ddelta,
            )
        )
    return entries


def _detect_crop_for_node(
    video_path: Path,
    sync_entries: list[SyncEntry],
    yolo_model: Any,
    yolo_conf: float,
    yolo_iou: float,
    yolo_device: str | None,
    yolo_samples_per_camera: int,
) -> tuple[int, int, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"failed to read first frame: {video_path}")
    img_h, img_w = frame.shape[:2]
    cap.release()

    if not sync_entries:
        return _compute_square_crop([0.0, 0.0, float(img_w - 1), float(img_h - 1)], img_w, img_h)

    sample_indices = np.linspace(
        0,
        len(sync_entries) - 1,
        num=min(yolo_samples_per_camera, len(sync_entries)),
        dtype=int,
    )
    chosen_bboxes: list[list[float]] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    for si in sample_indices.tolist():
        entry = sync_entries[si]
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(entry.color_frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            boxes = predict_human_bboxes(
                image_path=tmp_path,
                model=yolo_model,
                conf=yolo_conf,
                iou=yolo_iou,
                device=yolo_device,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if boxes:
            best = max(boxes, key=lambda b: b["score"])
            chosen_bboxes.append([float(x) for x in best["bbox"]])
    cap.release()

    if not chosen_bboxes:
        return _compute_square_crop([0.0, 0.0, float(img_w - 1), float(img_h - 1)], img_w, img_h)

    arr = np.array(chosen_bboxes, dtype=np.float32)
    bbox_union = [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max())]
    return _compute_square_crop(bbox_union, img_w, img_h)


def _rgb_crop_to_depth_crop(
    rgb_crop: tuple[int, int, int, int, int],
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
) -> tuple[int, int, int, int, int]:
    rx0, ry0, rx1, ry1, _ = rgb_crop
    rgb_w, rgb_h = rgb_size
    depth_w, depth_h = depth_size
    sx = float(depth_w) / float(rgb_w)
    sy = float(depth_h) / float(rgb_h)
    dx0 = rx0 * sx
    dy0 = ry0 * sy
    dx1 = rx1 * sx
    dy1 = ry1 * sy
    return _compute_square_crop([dx0, dy0, dx1, dy1], depth_w, depth_h)


class DepthReader:
    def __init__(self, path: Path, width: int = 512, height: int = 424) -> None:
        self.path = path
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 2
        self._fp = path.open("rb")

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        if frame_idx < 0:
            return None
        self._fp.seek(frame_idx * self._frame_bytes, os.SEEK_SET)
        raw = self._fp.read(self._frame_bytes)
        if len(raw) != self._frame_bytes:
            return None
        # Panoptic MATLAB reference:
        # im = reshape(data1, 512, 424)'; im = im(:, end:-1:1);
        # To match MATLAB reshape semantics in NumPy, use order='F'.
        arr = np.frombuffer(raw, dtype=np.uint16)
        arr = np.reshape(arr, (self.width, self.height), order="F").T[:, ::-1]
        return arr

    def close(self) -> None:
        self._fp.close()


def _compute_depth_crop_from_samples(
    depth_reader: DepthReader,
    sync_entries: list[SyncEntry],
    samples_per_camera: int,
) -> tuple[int, int, int, int, int] | None:
    if not sync_entries:
        return None
    sample_indices = np.linspace(
        0,
        len(sync_entries) - 1,
        num=min(samples_per_camera, len(sync_entries)),
        dtype=int,
    )
    min_x = None
    min_y = None
    max_x = None
    max_y = None
    for si in sample_indices.tolist():
        entry = sync_entries[si]
        depth = depth_reader.read_frame(entry.depth_frame_idx)
        if depth is None:
            continue
        ys, xs = np.where(depth > 0)
        if ys.size == 0:
            continue
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        min_x = x0 if min_x is None else min(min_x, x0)
        min_y = y0 if min_y is None else min(min_y, y0)
        max_x = x1 if max_x is None else max(max_x, x1)
        max_y = y1 if max_y is None else max(max_y, y1)
    if min_x is None:
        return None
    return _compute_square_crop(
        [float(min_x), float(min_y), float(max_x), float(max_y)],
        512,
        424,
    )


def _process_sequence(
    paths: SequencePaths,
    out_root: Path,
    rgb_out_size: tuple[int, int],
    depth_out_size: tuple[int, int],
    max_sync_delta_ms: float,
    max_body_frames: int | None,
    yolo_model: Any,
    yolo_conf: float,
    yolo_iou: float,
    yolo_device: str | None,
    yolo_samples_per_camera: int,
) -> dict[str, Any]:
    with paths.sync_path.open("r", encoding="utf-8") as f:
        sync_data = json.load(f)
    with paths.calib_path.open("r", encoding="utf-8") as f:
        calib_data = json.load(f)
    with paths.panoptic_calib_path.open("r", encoding="utf-8") as f:
        panoptic_calib_data = json.load(f)

    if "kinect" not in sync_data or "color" not in sync_data["kinect"] or "depth" not in sync_data["kinect"]:
        raise ValueError(f"invalid ksynctables format: {paths.sync_path}")

    body_frames = _load_body_frames(paths.body_dir, max_body_frames=max_body_frames)
    body_lookup = {bf.body_frame_id: bf for bf in body_frames}

    out_seq = out_root / paths.seq_name
    out_meta = out_seq / "meta"
    out_gt3d = out_seq / "gt3d"
    out_rgb = out_seq / "rgb"
    out_depth = out_seq / "depth"
    out_meta.mkdir(parents=True, exist_ok=True)
    out_gt3d.mkdir(parents=True, exist_ok=True)
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)

    sync_map: dict[str, list[dict[str, Any]]] = {}
    crop_params: dict[str, dict[str, int]] = {}
    total_written_rgb = 0

    for node_name, video_path in sorted(paths.video_paths.items()):
        color_node = sync_data["kinect"]["color"].get(node_name)
        depth_node = sync_data["kinect"]["depth"].get(node_name)
        if not isinstance(color_node, dict) or not isinstance(depth_node, dict):
            continue
        color_univ = np.array(color_node.get("univ_time", []), dtype=np.float64)
        depth_univ = np.array(depth_node.get("univ_time", []), dtype=np.float64)
        if color_univ.size == 0 or depth_univ.size == 0:
            continue

        entries = _build_sync_entries_for_node(
            body_frames=body_frames,
            color_univ_times=color_univ,
            depth_univ_times=depth_univ,
            max_sync_delta_ms=max_sync_delta_ms,
        )
        if not entries:
            continue

        rx0, ry0, rx1, ry1, rsize = _detect_crop_for_node(
            video_path=video_path,
            sync_entries=entries,
            yolo_model=yolo_model,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_device=yolo_device,
            yolo_samples_per_camera=yolo_samples_per_camera,
        )
        crop_params[node_name] = {"x0": rx0, "y0": ry0, "x1": rx1, "y1": ry1, "size": rsize}
        dx0, dy0, dx1, dy1, dsize = _rgb_crop_to_depth_crop(
            (rx0, ry0, rx1, ry1, rsize),
            rgb_size=(1920, 1080),
            depth_size=(512, 424),
        )
        crop_params[f"{node_name}_depth"] = {"x0": dx0, "y0": dy0, "x1": dx1, "y1": dy1, "size": dsize}

        cam_out = out_rgb / node_name.lower().replace("kinectnode", "kinect_")
        depth_out = out_depth / node_name.lower().replace("kinectnode", "kinect_")
        cam_out.mkdir(parents=True, exist_ok=True)
        depth_out.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video for writing pass: {video_path}")
        depth_reader = DepthReader(paths.depth_paths[node_name])
        node_sync_rows: list[dict[str, Any]] = []
        for entry in entries:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(entry.color_frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            depth_frame = depth_reader.read_frame(entry.depth_frame_idx)
            if depth_frame is None:
                continue
            rgb_crop = _crop_with_padding(frame, rx0, ry0, rx1, ry1, pad_value=(0, 0, 0))
            rgb_crop = cv2.resize(rgb_crop, rgb_out_size, interpolation=cv2.INTER_LINEAR)
            depth_crop = _crop_with_padding(depth_frame, dx0, dy0, dx1, dy1, pad_value=0)
            depth_crop = cv2.resize(depth_crop, depth_out_size, interpolation=cv2.INTER_NEAREST)
            out_name = f"{entry.body_frame_id:08d}.jpg"
            out_rgb_path = cam_out / out_name
            out_depth_path = depth_out / f"{entry.body_frame_id:08d}.png"
            cv2.imwrite(str(out_rgb_path), rgb_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(str(out_depth_path), depth_crop)
            total_written_rgb += 1

            node_sync_rows.append(
                {
                    "body_frame_id": entry.body_frame_id,
                    "body_univ_time": entry.body_univ_time,
                    "color_frame_idx": entry.color_frame_idx,
                    "color_univ_time": entry.color_univ_time,
                    "depth_frame_idx": entry.depth_frame_idx,
                    "depth_univ_time": entry.depth_univ_time,
                    "color_delta_ms": entry.color_delta_ms,
                    "depth_delta_ms": entry.depth_delta_ms,
                    "rgb_path": str(out_rgb_path.relative_to(out_seq)),
                    "depth_path": str(out_depth_path.relative_to(out_seq)),
                }
            )
        cap.release()
        depth_reader.close()
        sync_map[node_name] = node_sync_rows

    if not sync_map:
        raise RuntimeError(f"no synchronized camera streams produced for {paths.seq_name}")

    sensors = calib_data.get("sensors")
    if not isinstance(sensors, list) or len(sensors) < 10:
        raise ValueError(f"invalid sensors in calibration: {paths.calib_path}")
    panoptic_cameras = panoptic_calib_data.get("cameras")
    if not isinstance(panoptic_cameras, list) or not panoptic_cameras:
        raise ValueError(f"invalid panoptic cameras in calibration: {paths.panoptic_calib_path}")
    panoptic_camera_map = {}
    for cam in panoptic_cameras:
        name = cam.get("name")
        if not isinstance(name, str):
            continue
        panoptic_camera_map[name] = cam

    cameras_out: dict[str, dict[str, Any]] = {}
    for node_name in sorted(sync_map.keys()):
        node_idx = int(node_name.replace("KINECTNODE", "")) - 1
        if node_idx < 0 or node_idx >= len(sensors):
            raise ValueError(f"node index out of calibration range: {node_name}")
        sensor = sensors[node_idx]
        panoptic_cam_name = f"50_{node_idx + 1:02d}"
        if panoptic_cam_name not in panoptic_camera_map:
            raise ValueError(
                f"missing panoptic camera {panoptic_cam_name} in {paths.panoptic_calib_path}"
            )
        panoptic_cam = panoptic_camera_map[panoptic_cam_name]
        r_world_to_color = np.array(panoptic_cam["R"], dtype=np.float64)
        t_world_to_color_cm = np.array(panoptic_cam["t"], dtype=np.float64).reshape(3, 1)
        if r_world_to_color.shape != (3, 3):
            raise ValueError(
                f"invalid R shape for {panoptic_cam_name} in {paths.panoptic_calib_path}: "
                f"{r_world_to_color.shape}"
            )

        rgb_crop = crop_params[node_name]
        depth_crop = crop_params[f"{node_name}_depth"]
        k_color = np.array(sensor["K_color"], dtype=np.float64)
        k_depth = np.array(sensor["K_depth"], dtype=np.float64)
        k_color_new = _adjust_camera_intrinsic_for_crop(
            k_color,
            rgb_crop["x0"],
            rgb_crop["y0"],
            rgb_crop["size"],
            rgb_out_size,
        )
        k_depth_new = _adjust_camera_intrinsic_for_crop(
            k_depth,
            depth_crop["x0"],
            depth_crop["y0"],
            depth_crop["size"],
            depth_out_size,
        )
        cam_key = node_name.lower().replace("kinectnode", "kinect_")
        extrinsic_world_to_color = np.hstack((r_world_to_color, t_world_to_color_cm))
        cameras_out[cam_key] = {
            "node": node_name,
            "K_color": k_color_new.tolist(),
            "K_depth": k_depth_new.tolist(),
            "M_color": sensor["M_color"],
            "M_depth": sensor["M_depth"],
            "M_world2sensor": sensor["M_world2sensor"],
            "extrinsic_world_to_color": extrinsic_world_to_color.tolist(),
            "extrinsic_world_to_color_unit": "cm",
            "extrinsic_world_to_color_source": f"{paths.panoptic_calib_path.name}:{panoptic_cam_name}",
            "distCoeffs_color": sensor["distCoeffs_color"],
            "distCoeffs_depth": sensor["distCoeffs_depth"],
            "color_width": int(rgb_out_size[0]),
            "color_height": int(rgb_out_size[1]),
            "depth_width": int(depth_out_size[0]),
            "depth_height": int(depth_out_size[1]),
        }

    # Store gt3d once per body frame, independent of camera stream.
    for body_frame_id, bf in body_lookup.items():
        gt = np.array(bf.joints19, dtype=np.float32).reshape(19, 4)
        np.save(out_gt3d / f"{body_frame_id:08d}.npy", gt.astype(np.float16))

    with (out_meta / "sync_map.json").open("w", encoding="utf-8") as f:
        json.dump(sync_map, f, indent=2)
    with (out_meta / "crop_params.json").open("w", encoding="utf-8") as f:
        json.dump(crop_params, f, indent=2)
    with (out_meta / "cameras_kinect_cropped.json").open("w", encoding="utf-8") as f:
        json.dump(cameras_out, f, indent=2)
    with (out_meta / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sequence": paths.seq_name,
                "rgb_out_size": {"w": rgb_out_size[0], "h": rgb_out_size[1]},
                "depth_out_size": {"w": depth_out_size[0], "h": depth_out_size[1]},
                "max_sync_delta_ms": max_sync_delta_ms,
                "num_body_frames": len(body_frames),
                "num_camera_streams": len(sync_map),
                "num_written_rgb_frames": total_written_rgb,
                "cameras_file": "meta/cameras_kinect_cropped.json",
                "camera_extrinsics_self_contained": True,
            },
            f,
            indent=2,
        )

    return {
        "sequence": paths.seq_name,
        "status": "ok",
        "num_body_frames": len(body_frames),
        "num_camera_streams": len(sync_map),
        "num_written_rgb_frames": total_written_rgb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Panoptic Kinoptic single-actor sequences.")
    parser.add_argument("--root-dir", type=Path, default=Path("/data/shared/panoptic-toolbox"))
    parser.add_argument("--out-dir", type=Path, default=Path("/opt/data/panoptic_kinoptic_single_actor_cropped"))
    parser.add_argument("--sequences", type=str, default=None, help="Comma-separated sequence names.")
    parser.add_argument("--sequence-list", type=Path, default=None, help="Text file with one sequence name per line.")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-body-frames", type=int, default=None, help="Limit processed body frames per sequence.")
    parser.add_argument("--rgb-w", type=int, default=224)
    parser.add_argument("--rgb-h", type=int, default=224)
    parser.add_argument("--depth-w", type=int, default=224)
    parser.add_argument("--depth-h", type=int, default=224)
    parser.add_argument("--max-sync-delta-ms", type=float, default=25.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-device", type=str, default=None)
    parser.add_argument("--yolo-samples-per-camera", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rgb_w <= 0 or args.rgb_h <= 0:
        raise ValueError("rgb size must be positive")
    if args.depth_w <= 0 or args.depth_h <= 0:
        raise ValueError("depth size must be positive")
    if args.max_sync_delta_ms <= 0:
        raise ValueError("--max-sync-delta-ms must be > 0")
    if args.yolo_samples_per_camera <= 0:
        raise ValueError("--yolo-samples-per-camera must be > 0")
    if YOLO is None:
        raise ImportError("ultralytics is required for Panoptic Kinoptic preprocessing")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seqs = _parse_sequence_names(
        root_dir=args.root_dir,
        sequences_csv=args.sequences,
        sequence_list=args.sequence_list,
        max_sequences=args.max_sequences,
    )
    if not seqs:
        raise ValueError("no sequences selected")

    yolo_model = YOLO(args.yolo_weights)
    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for seq_name in tqdm(seqs, desc="Preprocess sequences"):
        try:
            paths = _discover_sequence_paths(args.root_dir, seq_name)
            summary = _process_sequence(
                paths=paths,
                out_root=args.out_dir,
                rgb_out_size=(args.rgb_w, args.rgb_h),
                depth_out_size=(args.depth_w, args.depth_h),
                max_sync_delta_ms=args.max_sync_delta_ms,
                max_body_frames=args.max_body_frames,
                yolo_model=yolo_model,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                yolo_device=args.yolo_device,
                yolo_samples_per_camera=args.yolo_samples_per_camera,
            )
            results.append(summary)
            print(
                f"[panoptic-preprocess] {seq_name}: ok, body_frames={summary['num_body_frames']}, "
                f"camera_streams={summary['num_camera_streams']}, rgb_frames={summary['num_written_rgb_frames']}"
            )
        except Exception as exc:
            msg = str(exc)
            failures.append((seq_name, msg))
            print(f"[panoptic-preprocess] {seq_name}: FAILED: {msg}")
            if not args.continue_on_error:
                raise

    print(f"[panoptic-preprocess] completed. success={len(results)} failed={len(failures)}")
    if failures:
        for seq_name, msg in failures:
            print(f"[panoptic-preprocess] failed: {seq_name}: {msg}")


if __name__ == "__main__":
    main()
