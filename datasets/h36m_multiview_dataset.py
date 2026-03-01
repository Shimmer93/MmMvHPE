import os.path as osp
import cv2
import json
import numpy as np
import re
from glob import glob
from typing import List, Sequence

from datasets.base_dataset import BaseDataset


H36M_JOINTS_TO_REMOVE = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]


class H36MMultiViewDataset(BaseDataset):
    TRAIN_SUBJECTS = {1, 5, 6, 7, 8}
    TEST_SUBJECTS = {9, 11}

    def __init__(
        self,
        data_root: str = "/opt/data/h36m_preprocessed",
        unit: str = "m",
        pipeline: List[dict] = [],
        split: str = "train",
        cameras: Sequence[str] = ("01", "02", "03", "04"),
        seq_len: int = 27,
        seq_step: int = 1,
        pad_seq: bool = False,
        causal: bool = False,
        return_keypoints_sequence: bool = True,
        load_rgb: bool = True,
    ):
        super().__init__(pipeline=pipeline)
        self.data_root = data_root
        self.rgb_root = osp.join(data_root, "rgb")
        self.gt_root = osp.join(data_root, "gt3d")
        self.unit = unit
        self.split = split
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.causal = causal
        self.pad_seq = pad_seq
        self.cameras = list(cameras)
        self.return_keypoints_sequence = return_keypoints_sequence
        self.load_rgb = load_rgb
        self.allowed_subjects = self._get_allowed_subjects()

        if unit not in {"mm", "m"}:
            self.unit = "m"

        self._frame_re = re.compile(
            r"s_(\d+)_act_(\d+)_subact_(\d+)_ca_(\d+)_([0-9]+)\.jpg"
        )

        self.rgb_size = self._infer_rgb_size()
        self.intrinsic_scale = (
            self.rgb_size[0] / 1000.0,
            self.rgb_size[1] / 1000.0,
        )
        self.camera_params = self._load_camera_params()
        self.data_list = self._build_dataset()

    def _get_allowed_subjects(self):
        if self.split in {"train"}:
            return self.TRAIN_SUBJECTS
        if self.split in {"test", "val", "valid", "validation"}:
            return self.TEST_SUBJECTS
        return None

    def _infer_rgb_size(self):
        rgb_files = glob(osp.join(self.rgb_root, "*.jpg"))
        if not rgb_files:
            return (1000, 1000)
        frame = cv2.imread(rgb_files[0], cv2.IMREAD_COLOR)
        if frame is None:
            return (1000, 1000)
        h, w = frame.shape[:2]
        return (w, h)

    def _load_camera_params(self):
        camera_path = osp.join(self.data_root, "camera-parameters.json")
        if not osp.exists(camera_path):
            return {}
        with open(camera_path, "r") as f:
            json_data = json.load(f)

        camera_id_map = {
            "54138969": "01",
            "55011271": "02",
            "58860488": "03",
            "60457274": "04",
        }
        camera_params = {}
        for subject in json_data.get("extrinsics", {}):
            subject_id = int(subject[1:])
            for cam_name, cam_label in camera_id_map.items():
                if cam_name not in json_data["extrinsics"][subject]:
                    continue
                extrinsic_data = json_data["extrinsics"][subject][cam_name]
                R = np.array(extrinsic_data["R"], dtype=np.float32)
                t = np.array(extrinsic_data["t"], dtype=np.float32).flatten()

                intrinsic_data = json_data["intrinsics"][cam_name]
                calib_matrix = np.array(intrinsic_data["calibration_matrix"], dtype=np.float32)
                sx, sy = self.intrinsic_scale
                fx = calib_matrix[0, 0] * sx
                fy = calib_matrix[1, 1] * sy
                cx = calib_matrix[0, 2] * sx
                cy = calib_matrix[1, 2] * sy
                distortion = np.array(intrinsic_data["distortion"], dtype=np.float32)

                camera_params[(subject_id, cam_label)] = {
                    "R": R,
                    "t": t.reshape(3, 1),
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "k": distortion[:3],
                    "p": distortion[3:5],
                    "name": cam_name,
                }
        return camera_params

    def _build_dataset(self):
        data_list = []
        rgb_files = glob(osp.join(self.rgb_root, "*.jpg"))

        frame_map = {}
        for fn in rgb_files:
            m = self._frame_re.match(osp.basename(fn))
            if not m:
                continue
            subject_id, action_id, subaction_id, cam_id, frame_idx = m.groups()
            key = (int(subject_id), int(action_id), int(subaction_id), cam_id)
            frame_map.setdefault(key, {})[int(frame_idx)] = fn

        ref_cam = self.cameras[0]
        keys = [k for k in frame_map.keys() if k[3] == ref_cam]
        for subject_id, action_id, subaction_id, _ in keys:
            if self.allowed_subjects is not None and subject_id not in self.allowed_subjects:
                continue
            frame_indices = sorted(frame_map[(subject_id, action_id, subaction_id, ref_cam)].keys())
            if len(frame_indices) < self.seq_len:
                continue

            for start in range(0, len(frame_indices) - self.seq_len + 1, self.seq_step):
                window = frame_indices[start : start + self.seq_len]
                if any(
                    any(idx not in frame_map.get((subject_id, action_id, subaction_id, cam), {}) for idx in window)
                    for cam in self.cameras
                ):
                    continue
                data_list.append({
                    "subject_id": subject_id,
                    "subject": f"S{subject_id}",
                    "action": action_id,
                    "subaction": subaction_id,
                    "frame_indices": window,
                })
        return data_list

    def __len__(self):
        return len(self.data_list)

    def _load_rgb_frame(self, basename):
        frame_path = osp.join(self.rgb_root, basename + ".jpg")
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"RGB frame not found: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _load_gt_frame(self, basename):
        gt_path = osp.join(self.gt_root, basename + ".npy")
        joints = np.load(gt_path)
        return joints

    def __getitem__(self, index):
        data_info = self.data_list[index]
        rgb_views = [] if self.load_rgb else None
        gt_keypoints_by_view = []
        camera_list = []

        for cam_label in self.cameras:
            frames = [] if self.load_rgb else None
            pose_seq = []
            for frame_idx in data_info["frame_indices"]:
                basename = (
                    f"s_{str(data_info['subject_id']).zfill(2)}_act_{str(data_info['action']).zfill(2)}_"
                    f"subact_{str(data_info['subaction']).zfill(2)}_ca_{cam_label}_"
                    f"{int(frame_idx):06d}"
                )
                if self.load_rgb:
                    frames.append(self._load_rgb_frame(basename))
                pose_seq.append(self._load_gt_frame(basename))
            if self.load_rgb:
                rgb_views.append(frames)
            gt_keypoints_by_view.append(np.stack(pose_seq, axis=0))

            cam_params = self.camera_params.get((data_info["subject_id"], cam_label), None)
            if cam_params is not None:
                T_scaled = cam_params["t"] / 1000.0 if self.unit == "m" else cam_params["t"]
                camera_list.append({
                    "intrinsic": np.array(
                        [[cam_params["fx"], 0.0, cam_params["cx"]],
                         [0.0, cam_params["fy"], cam_params["cy"]],
                         [0.0, 0.0, 1.0]],
                        dtype=np.float32,
                    ),
                    "extrinsic": np.hstack((cam_params["R"], T_scaled.reshape(3, 1))).astype(np.float32),
                    "camera_id": cam_label,
                })
            else:
                camera_list.append(None)

        if self.load_rgb:
            rgb_views = np.stack(rgb_views, axis=0)
        gt_keypoints_by_view = np.stack(gt_keypoints_by_view, axis=0)

        # Use a reference camera for GT to avoid averaging across camera coordinates.
        # Camera-specific 3D poses are in each camera's coordinate frame.
        ref_view = 0
        if self.causal:
            gt_keypoints = gt_keypoints_by_view[ref_view, -1]
        else:
            gt_keypoints = gt_keypoints_by_view[ref_view, self.seq_len // 2]

        sample = {
            "sample_id": f"{data_info['subject']}_{data_info['action']}_{data_info['subaction']}_{data_info['frame_indices'][0]}",
            "modalities": ["rgb"],
            "rgb_cameras": camera_list,
            "gt_keypoints": gt_keypoints,
            "gt_keypoints_by_view": gt_keypoints_by_view if self.return_keypoints_sequence else None,
            "cameras": self.cameras,
        }
        if self.load_rgb:
            sample["input_rgb"] = rgb_views

        sample = self.pipeline(sample)
        return sample
