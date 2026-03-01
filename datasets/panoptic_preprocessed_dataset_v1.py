import json
import os.path as osp
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from datasets.base_dataset import BaseDataset


class PanopticPreprocessedDatasetV1(BaseDataset):
    """Panoptic Kinoptic preprocessed dataset with HuMMan-v3-compatible sample keys.

    Expected layout per sequence under data_root:
      <seq>/rgb/kinect_X/*.jpg
      <seq>/depth/kinect_X/*.png
      <seq>/gt3d/*.npy
      <seq>/meta/sync_map.json
      <seq>/meta/cameras_kinect_cropped.json
      <seq>/meta/manifest.json
    """

    REQUIRED_REL_PATHS = (
        "meta/sync_map.json",
        "meta/cameras_kinect_cropped.json",
        "meta/manifest.json",
        "gt3d",
    )

    def __init__(
        self,
        data_root: str = "/opt/data/panoptic_kinoptic_single_actor_cropped",
        unit: str = "m",
        pipeline: List[dict] = [],
        split: str = "train",
        split_config: Optional[str] = None,
        split_to_use: str = "random_split",
        test_mode: bool = False,
        modality_names: Sequence[str] = ("rgb", "depth"),
        rgb_cameras: Optional[Sequence[str]] = None,
        depth_cameras: Optional[Sequence[str]] = None,
        rgb_cameras_per_sample: int = 1,
        depth_cameras_per_sample: int = 1,
        lidar_cameras_per_sample: int = 1,
        seq_len: int = 5,
        seq_step: int = 1,
        pad_seq: bool = False,
        causal: bool = False,
        use_all_pairs: bool = False,
        max_samples: Optional[int] = None,
        colocated: bool = False,
        convert_depth_to_lidar: bool = False,
        skeleton_only: bool = True,
        return_keypoints_sequence: bool = False,
        return_smpl_sequence: bool = False,
        sequence_allowlist: Optional[Sequence[str]] = None,
        sequence_list_file: Optional[str] = None,
        strict_validation: bool = True,
        random_seed: int = 0,
        gt_unit: str = "cm",
        output_num_joints: int = 19,
        apply_to_new_world: bool = False,
        remove_root_rotation: bool = False,
        panoptic_toolbox_root: Optional[str] = None,
        use_panoptic_calibration_extrinsics: bool = False,
    ):
        super().__init__(pipeline=pipeline)
        self.data_root = osp.abspath(osp.expanduser(data_root))
        self.unit = unit
        self.split = split
        self.split_config = split_config
        self.split_to_use = split_to_use
        self.test_mode = bool(test_mode)
        self.modality_names = [str(m).lower() for m in modality_names]
        self.seq_len = int(seq_len)
        self.seq_step = int(seq_step)
        self.pad_seq = bool(pad_seq)
        self.causal = bool(causal)
        self.use_all_pairs = bool(use_all_pairs)
        self.max_samples = max_samples
        self.colocated = bool(colocated)
        self.convert_depth_to_lidar = bool(convert_depth_to_lidar)
        self.skeleton_only = bool(skeleton_only)
        self.return_keypoints_sequence = bool(return_keypoints_sequence)
        self.return_smpl_sequence = bool(return_smpl_sequence)
        self.strict_validation = bool(strict_validation)
        self.random_seed = int(random_seed)
        self.gt_unit = str(gt_unit).lower()
        self.output_num_joints = int(output_num_joints)
        self.apply_to_new_world = bool(apply_to_new_world)
        self.remove_root_rotation = bool(remove_root_rotation)
        self.panoptic_toolbox_root = (
            osp.abspath(osp.expanduser(panoptic_toolbox_root))
            if panoptic_toolbox_root is not None
            else None
        )
        if self.remove_root_rotation and not self.apply_to_new_world:
            raise ValueError(
                "remove_root_rotation=True requires apply_to_new_world=True for PanopticPreprocessedDatasetV1."
            )
        self.use_panoptic_calibration_extrinsics = bool(use_panoptic_calibration_extrinsics)

        if self.gt_unit not in {"m", "cm", "mm"}:
            raise ValueError(f"Unsupported gt_unit={gt_unit}. Expected one of {{'m','cm','mm'}}.")
        if self.unit not in {"m", "mm"}:
            raise ValueError(f"Unsupported unit={unit}. Expected one of {{'m','mm'}}.")
        if self.output_num_joints <= 0:
            raise ValueError(f"output_num_joints must be > 0, got {self.output_num_joints}")

        self.rgb_cameras = self._normalize_camera_list(rgb_cameras)
        self.depth_cameras = self._normalize_camera_list(depth_cameras)
        self.lidar_cameras = list(self.depth_cameras)
        self.rgb_cameras_per_sample = max(1, int(rgb_cameras_per_sample))
        self.depth_cameras_per_sample = max(1, int(depth_cameras_per_sample))
        self.lidar_cameras_per_sample = max(1, int(lidar_cameras_per_sample))

        user_sequences = set(self._load_sequence_allowlist(sequence_allowlist, sequence_list_file))
        self._rng = random.Random(self.random_seed)

        self.sequence_data = self._index_sequences(user_sequences=user_sequences)
        self.data_list = self._build_dataset()
        if self.max_samples is not None:
            if self.max_samples <= 0:
                self.data_list = []
            else:
                idxs = self._rng.sample(range(len(self.data_list)), min(len(self.data_list), int(self.max_samples)))
                self.data_list = [self.data_list[i] for i in idxs]

    @staticmethod
    def _normalize_camera_name(name: str) -> str:
        n = str(name).strip().lower()
        if not n.startswith("kinect_"):
            return n
        tail = n.split("_", 1)[1]
        try:
            return f"kinect_{int(tail):03d}"
        except ValueError:
            return n

    @classmethod
    def _normalize_camera_list(cls, cameras: Optional[Sequence[str]]) -> List[str]:
        if cameras is None:
            return []
        return [cls._normalize_camera_name(c) for c in cameras]

    @staticmethod
    def _sample_camera_names(rng: random.Random, camera_pool: List[str], num_samples: int) -> List[str]:
        if not camera_pool:
            return []
        if len(camera_pool) >= num_samples:
            return rng.sample(camera_pool, num_samples)
        out = list(camera_pool)
        while len(out) < num_samples:
            out.append(rng.choice(camera_pool))
        return out

    @staticmethod
    def _maybe_single(items):
        if len(items) == 1:
            return items[0]
        return items

    @staticmethod
    def _depth_to_lidar_frames(depth_frames, K, min_depth=1e-6):
        K_inv = np.linalg.inv(K)
        pc_seq = []
        for depth in depth_frames:
            h, w = depth.shape
            xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
            z = depth.reshape(-1)
            valid = z > min_depth
            pixels = np.stack([xmap.reshape(-1), ymap.reshape(-1), np.ones(h * w)], axis=0)
            rays = K_inv @ pixels
            cam_points = rays * z
            cam_points = cam_points[:, valid]
            pc_seq.append(cam_points.T.astype(np.float32))
        return pc_seq

    def _load_sequence_allowlist(
        self,
        sequence_allowlist: Optional[Sequence[str]],
        sequence_list_file: Optional[str],
    ) -> List[str]:
        names: List[str] = []
        if sequence_allowlist:
            names.extend(str(x).strip() for x in sequence_allowlist if str(x).strip())
        if sequence_list_file:
            p = Path(sequence_list_file).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"sequence_list_file not found: {p}")
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.split("#", 1)[0].strip()
                if line:
                    names.append(line)
        return sorted(set(names))

    def _list_available_sequences(self) -> List[str]:
        root = Path(self.data_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Panoptic preprocessed data_root not found: {root}")
        return sorted([p.name for p in root.iterdir() if p.is_dir()])

    @staticmethod
    def _split_sequence_tokens(seq_name: str) -> Tuple[str, str]:
        if "_" not in seq_name:
            return seq_name, ""
        subject, action = seq_name.split("_", 1)
        return subject, action

    def _resolve_split_selection(self, available_sequences: List[str]) -> Dict[str, Any]:
        if self.split_config is None:
            return {"sequences": list(available_sequences), "cameras": None}

        cfg_path = Path(self.split_config).expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"split_config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if self.split_to_use not in cfg:
            raise ValueError(f"split_to_use={self.split_to_use} not found in split_config={cfg_path}")
        entry = cfg[self.split_to_use]
        split_key = "val_dataset" if self.test_mode else "train_dataset"
        if split_key not in entry:
            raise ValueError(f"Missing key `{split_key}` under split `{self.split_to_use}` in {cfg_path}")

        target = entry[split_key] or {}
        camera_filter = target.get("cameras", None)
        normalized_cameras = (
            [self._normalize_camera_name(c) for c in camera_filter]
            if camera_filter is not None
            else None
        )

        # 1) Explicit sequence list has top priority for all split modes.
        explicit_sequences = target.get("sequences", None)
        if explicit_sequences:
            requested = [str(x) for x in explicit_sequences]
            missing = sorted(set(requested) - set(available_sequences))
            if missing:
                raise ValueError(
                    f"split_config references unknown sequences ({len(missing)}): {missing[:10]}"
                )
            return {"sequences": sorted(requested), "cameras": normalized_cameras}

        # 2) random_split fallback from sequence ratio.
        if self.split_to_use == "random_split":
            ratio = entry.get("ratio", None)
            if ratio is None:
                raise ValueError(
                    "split_config requires explicit `sequences` or top-level `ratio` for deterministic split fallback"
                )
            seed = int(entry.get("random_seed", self.random_seed))
            if not (0.0 < float(ratio) < 1.0):
                raise ValueError(f"Invalid ratio={ratio}. Expected 0 < ratio < 1.")

            seqs = sorted(available_sequences)
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(seqs)).tolist()
            split_idx = int(np.floor(float(ratio) * len(seqs)))
            train_ids = [seqs[i] for i in order[:split_idx]]
            val_ids = [seqs[i] for i in order[split_idx:]]
            chosen = train_ids if split_key == "train_dataset" else val_ids
            return {"sequences": sorted(chosen), "cameras": normalized_cameras}

        # 3) cross_* style filtering by subjects/actions (HuMMan-style config contract).
        subjects = target.get("subjects", None)
        actions = target.get("actions", None)
        subject_set = {str(s) for s in subjects} if subjects else None
        action_set = {str(a) for a in actions} if actions else None

        selected = []
        for seq_name in sorted(available_sequences):
            subject, action = self._split_sequence_tokens(seq_name)
            if subject_set is not None and subject not in subject_set:
                continue
            if action_set is not None and action not in action_set:
                continue
            selected.append(seq_name)

        if not selected:
            raise ValueError(
                f"Split `{self.split_to_use}` with `{split_key}` selected zero sequences. "
                "Check `subjects`/`actions`/`sequences` entries against Panoptic sequence names."
            )
        return {"sequences": selected, "cameras": normalized_cameras}

    def _camera_extrinsic(self, cam: Dict[str, Any], modality: str) -> np.ndarray:
        m_world2sensor = np.asarray(cam["M_world2sensor"], dtype=np.float32)
        if modality == "rgb":
            m_sensor2mod = np.asarray(cam["M_color"], dtype=np.float32)
        elif modality in {"depth", "lidar"}:
            m_sensor2mod = np.asarray(cam["M_depth"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported modality for camera extrinsic: {modality}")
        if m_world2sensor.shape != (4, 4) or m_sensor2mod.shape != (4, 4):
            raise ValueError("Camera matrices must be 4x4.")

        m_world2mod = m_sensor2mod @ m_world2sensor
        return np.asarray(m_world2mod[:3, :4], dtype=np.float32)

    def _camera_intrinsic(self, cam: Dict[str, Any], modality: str) -> np.ndarray:
        key = "K_color" if modality == "rgb" else "K_depth"
        k = np.asarray(cam[key], dtype=np.float32)
        if k.shape != (3, 3):
            raise ValueError(f"Invalid intrinsic shape for {key}: {k.shape}")
        return k

    def _index_sequences(self, user_sequences: set[str]) -> Dict[str, Dict[str, Any]]:
        available_sequences = self._list_available_sequences()
        split_selection = self._resolve_split_selection(available_sequences)
        split_sequences = split_selection["sequences"]
        split_cameras = split_selection["cameras"]

        if user_sequences:
            missing = sorted(user_sequences - set(available_sequences))
            if missing:
                raise ValueError(
                    f"User sequence allowlist references unknown sequences ({len(missing)}): {missing[:10]}"
                )
            selected_sequences = sorted(set(split_sequences) & user_sequences)
            if not selected_sequences:
                raise ValueError("No sequences remain after intersecting split selection and sequence_allowlist.")
        else:
            selected_sequences = split_sequences

        out: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        for seq_name in selected_sequences:
            seq_dir = Path(self.data_root) / seq_name
            try:
                for rel in self.REQUIRED_REL_PATHS:
                    p = seq_dir / rel
                    if not p.exists():
                        raise FileNotFoundError(f"missing required artifact: {p}")

                sync_path = seq_dir / "meta" / "sync_map.json"
                cam_path = seq_dir / "meta" / "cameras_kinect_cropped.json"
                gt_dir = seq_dir / "gt3d"

                with sync_path.open("r", encoding="utf-8") as f:
                    sync_map = json.load(f)
                with cam_path.open("r", encoding="utf-8") as f:
                    cameras = json.load(f)

                if not isinstance(sync_map, dict) or not sync_map:
                    raise ValueError(f"sync_map is empty or invalid: {sync_path}")
                if not isinstance(cameras, dict) or not cameras:
                    raise ValueError(f"camera metadata is empty or invalid: {cam_path}")

                rgb_by_cam: Dict[str, Dict[int, str]] = {}
                depth_by_cam: Dict[str, Dict[int, str]] = {}
                cam_name_to_raw: Dict[str, str] = {}

                for node_name, rows in sync_map.items():
                    if not isinstance(rows, list):
                        raise ValueError(f"sync_map[{node_name}] must be list, got {type(rows).__name__}")
                    for row in rows:
                        if not isinstance(row, dict):
                            raise ValueError(f"sync_map row must be dict, got {type(row).__name__}")
                        body_frame_id = int(row["body_frame_id"])
                        rgb_rel = row.get("rgb_path")
                        depth_rel = row.get("depth_path")
                        if not isinstance(rgb_rel, str) or not isinstance(depth_rel, str):
                            raise ValueError(f"sync_map row missing rgb_path/depth_path in sequence {seq_name}")

                        rgb_parts = Path(rgb_rel).parts
                        depth_parts = Path(depth_rel).parts
                        if len(rgb_parts) < 3 or len(depth_parts) < 3:
                            raise ValueError(f"Invalid rgb/depth path format in sync_map for sequence {seq_name}")

                        rgb_cam_raw = rgb_parts[1]
                        depth_cam_raw = depth_parts[1]
                        rgb_cam_norm = self._normalize_camera_name(rgb_cam_raw)
                        depth_cam_norm = self._normalize_camera_name(depth_cam_raw)
                        if rgb_cam_norm != depth_cam_norm:
                            raise ValueError(
                                f"RGB/depth camera mismatch in sync row for {seq_name}: {rgb_cam_raw} vs {depth_cam_raw}"
                            )
                        cam_name_to_raw[rgb_cam_norm] = rgb_cam_raw

                        rgb_abs = seq_dir / rgb_rel
                        depth_abs = seq_dir / depth_rel
                        if not rgb_abs.is_file() or not depth_abs.is_file():
                            raise FileNotFoundError(
                                f"Missing synchronized frame for {seq_name}: {rgb_abs} or {depth_abs}"
                            )

                        rgb_by_cam.setdefault(rgb_cam_norm, {})[body_frame_id] = str(rgb_abs)
                        depth_by_cam.setdefault(depth_cam_norm, {})[body_frame_id] = str(depth_abs)

                if "rgb" in self.modality_names and not rgb_by_cam:
                    raise ValueError(f"No RGB camera streams found for sequence {seq_name}")
                if "depth" in self.modality_names and not depth_by_cam:
                    raise ValueError(f"No depth camera streams found for sequence {seq_name}")

                gt_files = sorted(gt_dir.glob("*.npy"))
                if not gt_files:
                    raise FileNotFoundError(f"No gt3d files under {gt_dir}")
                gt_by_frame = {}
                for p in gt_files:
                    try:
                        frame_id = int(p.stem)
                    except ValueError as exc:
                        raise ValueError(f"Invalid gt3d filename (expected integer stem): {p.name}") from exc
                    gt_by_frame[frame_id] = str(p)

                # Camera-level filtering.
                rgb_cam_names = sorted(rgb_by_cam.keys())
                depth_cam_names = sorted(depth_by_cam.keys())
                if self.rgb_cameras:
                    rgb_cam_names = [c for c in rgb_cam_names if c in set(self.rgb_cameras)]
                if self.depth_cameras:
                    depth_cam_names = [c for c in depth_cam_names if c in set(self.depth_cameras)]
                if split_cameras:
                    rgb_cam_names = [c for c in rgb_cam_names if c in set(split_cameras)]
                    depth_cam_names = [c for c in depth_cam_names if c in set(split_cameras)]
                if self.colocated and "rgb" in self.modality_names and "depth" in self.modality_names:
                    common = sorted(set(rgb_cam_names) & set(depth_cam_names))
                    rgb_cam_names = common
                    depth_cam_names = common

                if "rgb" in self.modality_names and not rgb_cam_names:
                    raise ValueError(f"No RGB cameras left after filtering for sequence {seq_name}")
                if "depth" in self.modality_names and not depth_cam_names:
                    raise ValueError(f"No depth cameras left after filtering for sequence {seq_name}")

                common_ids = set(gt_by_frame.keys())
                if "rgb" in self.modality_names:
                    for cam in rgb_cam_names:
                        common_ids &= set(rgb_by_cam[cam].keys())
                if "depth" in self.modality_names:
                    for cam in depth_cam_names:
                        common_ids &= set(depth_by_cam[cam].keys())

                frame_ids = sorted(common_ids)
                if len(frame_ids) < self.seq_len and not self.pad_seq:
                    raise ValueError(
                        f"Not enough synchronized frames in {seq_name}: {len(frame_ids)} < seq_len={self.seq_len}"
                    )
                if not frame_ids:
                    raise ValueError(f"No synchronized frame IDs remain after validation for sequence {seq_name}")

                out[seq_name] = {
                    "seq_dir": str(seq_dir),
                    "rgb_by_cam": rgb_by_cam,
                    "depth_by_cam": depth_by_cam,
                    "gt_by_frame": gt_by_frame,
                    "frame_ids": frame_ids,
                    "rgb_cams": rgb_cam_names,
                    "depth_cams": depth_cam_names,
                    "cameras": cameras,
                    "cam_name_to_raw": cam_name_to_raw,
                    "panoptic_extrinsics": self._load_panoptic_extrinsics(seq_name),
                }
            except Exception as exc:
                errors.append(f"{seq_name}: {exc}")

        if errors:
            msg = "Panoptic sequence validation failed:\n" + "\n".join(errors[:20])
            if self.strict_validation:
                raise ValueError(msg)
            print(f"[PanopticPreprocessedDatasetV1] WARNING: {msg}")

        if not out:
            raise ValueError("No valid sequences were indexed.")
        return out

    def _load_panoptic_extrinsics(self, seq_name: str) -> Dict[str, np.ndarray]:
        if not self.use_panoptic_calibration_extrinsics or self.panoptic_toolbox_root is None:
            return {}

        calib_path = Path(self.panoptic_toolbox_root) / seq_name / f"calibration_{seq_name}.json"
        if not calib_path.is_file():
            return {}

        with calib_path.open("r", encoding="utf-8") as f:
            calib = json.load(f)
        cameras = calib.get("cameras", [])
        if not isinstance(cameras, list):
            return {}

        # Panoptic calibration translations are in centimeters.
        if self.unit == "m":
            t_scale = 0.01
        elif self.unit == "mm":
            t_scale = 10.0
        else:
            t_scale = 1.0

        out: Dict[str, np.ndarray] = {}
        for cam in cameras:
            name = cam.get("name", "")
            if not isinstance(name, str) or not name.startswith("50_"):
                continue
            try:
                cam_idx = int(name.split("_", 1)[1])
            except ValueError:
                continue
            cam_name = self._normalize_camera_name(f"kinect_{cam_idx}")
            r = np.asarray(cam["R"], dtype=np.float32)
            t = np.asarray(cam["t"], dtype=np.float32).reshape(3, 1) * float(t_scale)
            if r.shape != (3, 3):
                continue
            out[cam_name] = np.hstack((r, t)).astype(np.float32)
        return out

    def _build_dataset(self) -> List[Dict[str, Any]]:
        data_list: List[Dict[str, Any]] = []
        for seq_name in sorted(self.sequence_data.keys()):
            seq_info = self.sequence_data[seq_name]
            frame_ids = seq_info["frame_ids"]
            rgb_cams = list(seq_info["rgb_cams"])
            depth_cams = list(seq_info["depth_cams"])
            lidar_cams = list(depth_cams)

            if len(frame_ids) < self.seq_len and self.pad_seq:
                starts = [0]
            else:
                starts = list(range(0, max(0, len(frame_ids) - self.seq_len + 1), self.seq_step))

            for start_idx in starts:
                if self.use_all_pairs:
                    rgb_pool = rgb_cams if rgb_cams else [None]
                    depth_pool = depth_cams if depth_cams else [None]
                    for rgb_cam in rgb_pool:
                        for depth_cam in depth_pool:
                            data_list.append(
                                {
                                    "seq_name": seq_name,
                                    "start_frame": start_idx,
                                    "num_frames": len(frame_ids),
                                    "rgb_camera": rgb_cam,
                                    "depth_camera": depth_cam,
                                    "lidar_camera": depth_cam,
                                    "rgb_cameras": list(rgb_cams),
                                    "depth_cameras": list(depth_cams),
                                    "lidar_cameras": list(lidar_cams),
                                }
                            )
                else:
                    data_list.append(
                        {
                            "seq_name": seq_name,
                            "start_frame": start_idx,
                            "num_frames": len(frame_ids),
                            "rgb_camera": None,
                            "depth_camera": None,
                            "lidar_camera": None,
                            "rgb_cameras": list(rgb_cams),
                            "depth_cameras": list(depth_cams),
                            "lidar_cameras": list(lidar_cams),
                        }
                    )
        return data_list

    def _load_gt_keypoints(self, seq_name: str, body_frame_id: int) -> np.ndarray:
        gt_path = self.sequence_data[seq_name]["gt_by_frame"][body_frame_id]
        gt = np.load(gt_path).astype(np.float32)
        if gt.ndim != 2 or gt.shape[1] < 3:
            raise ValueError(f"Invalid gt3d shape in {gt_path}: {gt.shape}")
        xyz = gt[:, :3]
        if self.gt_unit == "cm" and self.unit == "m":
            xyz = xyz / 100.0
        elif self.gt_unit == "mm" and self.unit == "m":
            xyz = xyz / 1000.0
        elif self.gt_unit == "m" and self.unit == "mm":
            xyz = xyz * 1000.0
        xyz = xyz.astype(np.float32)
        if xyz.shape[0] != self.output_num_joints:
            raise ValueError(
                f"GT joints ({xyz.shape[0]}) do not match output_num_joints ({self.output_num_joints}) "
                f"for seq={seq_name}, frame={body_frame_id}"
            )
        return xyz

    def _load_rgb_frames(self, seq_name: str, camera_name: str, frame_window: List[int]) -> List[np.ndarray]:
        rgb_map = self.sequence_data[seq_name]["rgb_by_cam"][camera_name]
        frames: List[np.ndarray] = []
        for body_frame_id in frame_window:
            frame = cv2.imread(rgb_map[body_frame_id], cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read RGB frame: {rgb_map[body_frame_id]}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames

    def _load_depth_frames(self, seq_name: str, camera_name: str, frame_window: List[int]) -> List[np.ndarray]:
        depth_map = self.sequence_data[seq_name]["depth_by_cam"][camera_name]
        frames: List[np.ndarray] = []
        for body_frame_id in frame_window:
            depth = cv2.imread(depth_map[body_frame_id], cv2.IMREAD_ANYDEPTH)
            if depth is None:
                depth = cv2.imread(depth_map[body_frame_id], cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"Failed to read depth frame: {depth_map[body_frame_id]}")
            depth = depth.astype(np.float32)
            if self.unit == "m":
                depth = depth / 1000.0
            frames.append(depth)
        return frames

    def _camera_params(self, seq_name: str, camera_name: str, modality: str) -> Dict[str, np.ndarray]:
        seq_info = self.sequence_data[seq_name]
        raw_name = seq_info["cam_name_to_raw"].get(camera_name, camera_name)
        if raw_name not in seq_info["cameras"]:
            raise KeyError(f"Camera {raw_name} not found in cropped cameras metadata for {seq_name}")
        cam = seq_info["cameras"][raw_name]
        if modality in {"rgb", "depth", "lidar"} and "extrinsic_world_to_color" in cam:
            ext = np.asarray(cam["extrinsic_world_to_color"], dtype=np.float32)
            if ext.shape != (3, 4):
                raise ValueError(
                    f"Invalid extrinsic_world_to_color shape for {seq_name}/{raw_name}: {ext.shape}"
                )
            ext_unit = str(cam.get("extrinsic_world_to_color_unit", "cm")).lower()
            ext = ext.copy()
            if ext_unit == "cm":
                if self.unit == "m":
                    ext[:, 3] *= 0.01
                elif self.unit == "mm":
                    ext[:, 3] *= 10.0
            elif ext_unit == "m":
                if self.unit == "mm":
                    ext[:, 3] *= 1000.0
            elif ext_unit == "mm":
                if self.unit == "m":
                    ext[:, 3] *= 0.001
            else:
                raise ValueError(
                    f"Unsupported extrinsic_world_to_color_unit={ext_unit} for {seq_name}/{raw_name}"
                )
            extrinsic = ext
        else:
            extrinsic = None
        ext_map = seq_info.get("panoptic_extrinsics", {})
        if extrinsic is None:
            if modality == "rgb" and camera_name in ext_map:
                extrinsic = ext_map[camera_name]
            elif modality in {"depth", "lidar"} and camera_name in ext_map:
                # Use color extrinsic for depth streams since RGB/depth were synchronized and cropped together.
                extrinsic = ext_map[camera_name]
            else:
                extrinsic = self._camera_extrinsic(cam, modality)
        return {
            "intrinsic": self._camera_intrinsic(cam, modality),
            "extrinsic": extrinsic,
        }

    @staticmethod
    def _world_to_new_world(points: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        pel = np.asarray(pelvis, dtype=np.float32).reshape(1, 3)
        return (pts - pel).astype(np.float32)

    @staticmethod
    def _world_to_new_world_rot(
        points: np.ndarray,
        pelvis: np.ndarray,
        r_new_to_world: np.ndarray,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        pel = np.asarray(pelvis, dtype=np.float32).reshape(1, 3)
        r = np.asarray(r_new_to_world, dtype=np.float32)
        if r.shape != (3, 3):
            raise ValueError(f"r_new_to_world must be (3,3), got {r.shape}")
        pts_new = (r.T @ (pts - pel).T).T
        return pts_new.reshape(points.shape).astype(np.float32)

    @staticmethod
    def _camera_to_new_world(
        camera: Dict[str, np.ndarray],
        pelvis: np.ndarray,
        r_new_to_world: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        out = dict(camera)
        ext = np.asarray(camera["extrinsic"], dtype=np.float32)
        if ext.shape != (3, 4):
            raise ValueError(f"Camera extrinsic must be (3,4), got {ext.shape}")
        r_wc = ext[:, :3]
        t_wc = ext[:, 3:4]
        pel = np.asarray(pelvis, dtype=np.float32).reshape(3, 1)
        if r_new_to_world is None:
            r_new_to_world = np.eye(3, dtype=np.float32)
        r_new_to_world = np.asarray(r_new_to_world, dtype=np.float32)
        if r_new_to_world.shape != (3, 3):
            raise ValueError(f"r_new_to_world must be (3,3), got {r_new_to_world.shape}")
        r_new = r_wc @ r_new_to_world
        t_new = r_wc @ pel + t_wc
        out["extrinsic"] = np.hstack((r_new, t_new)).astype(np.float32)
        return out

    @staticmethod
    def _normalize_vec(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= eps:
            raise ValueError("Cannot normalize near-zero vector while estimating root rotation.")
        return (v / n).astype(np.float32)

    def _estimate_root_rotation_from_joints19(self, keypoints: np.ndarray) -> np.ndarray:
        # Panoptic joints19 indices:
        # 0: Neck, 2: BodyCenter, 6: lHip, 12: rHip
        if keypoints.shape[0] <= 12:
            raise ValueError(
                f"Need Panoptic joints19 indices [0,2,6,12], got shape={keypoints.shape}."
            )
        neck = np.asarray(keypoints[0], dtype=np.float32)
        body = np.asarray(keypoints[2], dtype=np.float32)
        lhip = np.asarray(keypoints[6], dtype=np.float32)
        rhip = np.asarray(keypoints[12], dtype=np.float32)

        x_axis = self._normalize_vec(rhip - lhip)       # right direction
        y_seed = self._normalize_vec(neck - body)       # up direction
        z_axis = self._normalize_vec(np.cross(x_axis, y_seed))   # forward direction
        y_axis = self._normalize_vec(np.cross(z_axis, x_axis))   # re-orthogonalize up

        r_new_to_world = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
        det = float(np.linalg.det(r_new_to_world))
        if not np.isfinite(det) or abs(det) < 1e-5:
            raise ValueError(f"Invalid root rotation matrix estimated from joints19 (det={det}).")
        return r_new_to_world

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        data_info = self.data_list[index].copy()
        seq_name = data_info["seq_name"]
        seq_info = self.sequence_data[seq_name]
        frame_ids = seq_info["frame_ids"]

        selected_rgb: List[str] = []
        selected_depth: List[str] = []
        selected_lidar: List[str] = []

        if not self.use_all_pairs:
            if "rgb" in self.modality_names:
                selected_rgb = self._sample_camera_names(
                    self._rng,
                    list(data_info.get("rgb_cameras", [])),
                    self.rgb_cameras_per_sample,
                )
            if "depth" in self.modality_names:
                selected_depth = self._sample_camera_names(
                    self._rng,
                    list(data_info.get("depth_cameras", [])),
                    self.depth_cameras_per_sample,
                )
            if "lidar" in self.modality_names:
                selected_lidar = self._sample_camera_names(
                    self._rng,
                    list(data_info.get("lidar_cameras", [])),
                    self.lidar_cameras_per_sample,
                )
            if self.colocated and selected_rgb and selected_depth:
                common = sorted(list(set(selected_rgb) & set(selected_depth)))
                if common:
                    selected_rgb = self._sample_camera_names(self._rng, common, self.rgb_cameras_per_sample)
                    selected_depth = self._sample_camera_names(self._rng, common, self.depth_cameras_per_sample)
                    if "lidar" in self.modality_names:
                        selected_lidar = self._sample_camera_names(self._rng, common, self.lidar_cameras_per_sample)
        else:
            if "rgb" in self.modality_names and data_info.get("rgb_camera") is not None:
                selected_rgb = [data_info["rgb_camera"]]
            if "depth" in self.modality_names and data_info.get("depth_camera") is not None:
                selected_depth = [data_info["depth_camera"]]
            if "lidar" in self.modality_names and data_info.get("lidar_camera") is not None:
                selected_lidar = [data_info["lidar_camera"]]

        if not selected_rgb and "rgb" in self.modality_names:
            raise ValueError(f"No RGB camera selected for seq={seq_name}")
        if not selected_depth and "depth" in self.modality_names:
            raise ValueError(f"No depth camera selected for seq={seq_name}")

        # Build frame window by shared body_frame_id index.
        start = int(data_info["start_frame"])
        frame_window: List[int] = []
        for i in range(self.seq_len):
            idx = start + i
            if idx >= len(frame_ids):
                if not self.pad_seq:
                    break
                idx = len(frame_ids) - 1
            frame_window.append(frame_ids[idx])
        if not frame_window:
            raise ValueError(f"Empty frame window for seq={seq_name}, start={start}")

        if self.causal:
            gt_body_frame_id = frame_window[-1]
        else:
            gt_body_frame_id = frame_window[len(frame_window) // 2]

        gt_keypoints = self._load_gt_keypoints(seq_name, gt_body_frame_id)
        if gt_keypoints.shape[0] <= 2:
            raise ValueError(
                f"Panoptic joints19 requires BodyCenter at index 2, got shape={gt_keypoints.shape} "
                f"for seq={seq_name}, frame={gt_body_frame_id}"
            )
        # Panoptic joints19 order: 2 = BodyCenter (center of hips), used as pelvis/root.
        pelvis = np.asarray(gt_keypoints[2], dtype=np.float32)
        gt_keypoints = gt_keypoints.astype(np.float32)
        r_new_to_world = np.eye(3, dtype=np.float32)
        if self.apply_to_new_world and self.remove_root_rotation:
            r_new_to_world = self._estimate_root_rotation_from_joints19(gt_keypoints)
        if self.apply_to_new_world:
            if self.remove_root_rotation:
                gt_keypoints = self._world_to_new_world_rot(gt_keypoints, pelvis, r_new_to_world)
            else:
                gt_keypoints = self._world_to_new_world(gt_keypoints, pelvis)

        gt_smpl_params = np.zeros((82,), dtype=np.float32)
        gt_global_orient = np.zeros((3,), dtype=np.float32)

        primary_rgb = selected_rgb[0] if selected_rgb else None
        primary_depth = selected_depth[0] if selected_depth else None
        sample = {
            "sample_id": (
                f"{seq_name}_rgb_{primary_rgb}_depth_{primary_depth}_{gt_body_frame_id:08d}"
            ),
            "modalities": list(self.modality_names),
            "gt_keypoints": gt_keypoints,
            "gt_smpl_params": gt_smpl_params,
            "gt_global_orient": gt_global_orient,
            "gt_pelvis": pelvis,
            "seq_name": seq_name,
            "start_frame": start,
            "selected_cameras": {
                "rgb": list(selected_rgb),
                "depth": list(selected_depth),
                "lidar": list(selected_lidar),
            },
        }

        if self.return_keypoints_sequence:
            seq_kpts = [self._load_gt_keypoints(seq_name, fid) for fid in frame_window]
            seq_kpts = np.stack(seq_kpts, axis=0).astype(np.float32)
            if self.apply_to_new_world:
                if self.remove_root_rotation:
                    seq_kpts = self._world_to_new_world_rot(seq_kpts, pelvis, r_new_to_world)
                else:
                    seq_kpts = seq_kpts - pelvis.reshape(1, 1, 3)
            sample["gt_keypoints_seq"] = seq_kpts

        if self.return_smpl_sequence:
            sample["gt_smpl_params_seq"] = np.zeros((len(frame_window), 82), dtype=np.float32)

        if "rgb" in self.modality_names:
            rgb_frames_views = []
            rgb_cameras = []
            for cam in selected_rgb:
                if self.skeleton_only:
                    rgb_frames_views.append(self._load_rgb_frames(seq_name, cam, frame_window))
                rgb_cameras.append(self._camera_params(seq_name, cam, "rgb"))
            if self.skeleton_only and rgb_frames_views:
                sample["input_rgb"] = self._maybe_single(rgb_frames_views)
            if rgb_cameras:
                if self.apply_to_new_world:
                    rgb_cameras = [
                        self._camera_to_new_world(cam, pelvis, r_new_to_world if self.remove_root_rotation else None)
                        for cam in rgb_cameras
                    ]
                sample["rgb_camera"] = self._maybe_single(rgb_cameras)

        if "depth" in self.modality_names:
            depth_frames_views = []
            depth_cameras = []
            for cam in selected_depth:
                if self.skeleton_only:
                    depth_frames_views.append(self._load_depth_frames(seq_name, cam, frame_window))
                depth_cameras.append(self._camera_params(seq_name, cam, "depth"))

            if self.skeleton_only and depth_frames_views:
                sample["input_depth"] = self._maybe_single(depth_frames_views)

            converted_to_lidar = self.convert_depth_to_lidar and "lidar" not in self.modality_names
            if converted_to_lidar and depth_frames_views:
                lidar_frames_views = []
                for depth_frames, camera in zip(depth_frames_views, depth_cameras):
                    lidar_frames_views.append(self._depth_to_lidar_frames(depth_frames, camera["intrinsic"]))
                sample["input_lidar"] = self._maybe_single(lidar_frames_views)
                if "lidar" not in sample["modalities"]:
                    sample["modalities"].append("lidar")
                if "depth" in sample["modalities"]:
                    sample["modalities"].remove("depth")
                sample.pop("input_depth", None)
                sample["selected_cameras"]["lidar"] = list(selected_depth)
                if self.apply_to_new_world:
                    depth_cameras = [
                        self._camera_to_new_world(cam, pelvis, r_new_to_world if self.remove_root_rotation else None)
                        for cam in depth_cameras
                    ]
                sample["lidar_camera"] = self._maybe_single(depth_cameras)
            elif depth_cameras:
                if self.apply_to_new_world:
                    depth_cameras = [
                        self._camera_to_new_world(cam, pelvis, r_new_to_world if self.remove_root_rotation else None)
                        for cam in depth_cameras
                    ]
                sample["depth_camera"] = self._maybe_single(depth_cameras)

        sample = self.pipeline(sample)
        return sample
