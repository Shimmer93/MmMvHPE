from __future__ import annotations

from typing import Any

import numpy as np
import torch

from misc.skeleton import SMPLSkeleton


def sam3_cam_int_from_rgb_camera(rgb_camera: dict[str, Any]) -> torch.Tensor:
    intrinsic = np.asarray(rgb_camera["intrinsic"], dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected rgb intrinsic with shape (3,3), got {intrinsic.shape}.")
    return torch.from_numpy(intrinsic[None, ...].copy())


class SAM3ToHummanSMPL24Adapter:
    TARGET_NAMES = list(SMPLSkeleton.joint_names)

    def __init__(self) -> None:
        from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

        keypoint_info = mhr70_pose_info["keypoint_info"]
        self.source_name_to_idx = {
            str(meta["name"]).strip(): int(kpt_id) for kpt_id, meta in keypoint_info.items()
        }
        required = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_big_toe",
            "left_small_toe",
            "left_heel",
            "right_big_toe",
            "right_small_toe",
            "right_heel",
            "left_wrist",
            "right_wrist",
            "left_acromion",
            "right_acromion",
            "neck",
        ]
        missing = [name for name in required if name not in self.source_name_to_idx]
        if missing:
            raise ValueError(
                "SAM3 MHR70 metadata is missing joints required for HuMMan SMPL24 conversion: "
                + ", ".join(sorted(missing))
            )
        self.num_joints = 24
        self.pelvis_idx = 0

    def _src(self, joints: np.ndarray, name: str) -> np.ndarray:
        return joints[self.source_name_to_idx[name]]

    def adapt(self, sam_keypoints_3d: np.ndarray) -> np.ndarray:
        joints = np.asarray(sam_keypoints_3d, dtype=np.float32)
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError(f"SAM keypoints must have shape (J,3), got {joints.shape}.")

        left_hip = self._src(joints, "left_hip")
        right_hip = self._src(joints, "right_hip")
        pelvis = 0.5 * (left_hip + right_hip)
        neck = self._src(joints, "neck")
        spine1 = pelvis + (neck - pelvis) * (1.0 / 3.0)
        spine2 = pelvis + (neck - pelvis) * (2.0 / 3.0)
        spine3 = neck
        left_foot = np.mean(
            np.stack(
                [
                    self._src(joints, "left_ankle"),
                    self._src(joints, "left_big_toe"),
                    self._src(joints, "left_small_toe"),
                    self._src(joints, "left_heel"),
                ],
                axis=0,
            ),
            axis=0,
        )
        right_foot = np.mean(
            np.stack(
                [
                    self._src(joints, "right_ankle"),
                    self._src(joints, "right_big_toe"),
                    self._src(joints, "right_small_toe"),
                    self._src(joints, "right_heel"),
                ],
                axis=0,
            ),
            axis=0,
        )
        head = np.mean(
            np.stack(
                [
                    self._src(joints, "nose"),
                    self._src(joints, "left_eye"),
                    self._src(joints, "right_eye"),
                    self._src(joints, "left_ear"),
                    self._src(joints, "right_ear"),
                ],
                axis=0,
            ),
            axis=0,
        )
        left_hand = self._src(joints, "left_wrist")
        right_hand = self._src(joints, "right_wrist")

        out = np.stack(
            [
                pelvis,
                left_hip,
                right_hip,
                spine1,
                self._src(joints, "left_knee"),
                self._src(joints, "right_knee"),
                spine2,
                self._src(joints, "left_ankle"),
                self._src(joints, "right_ankle"),
                spine3,
                left_foot,
                right_foot,
                neck,
                self._src(joints, "left_acromion"),
                self._src(joints, "right_acromion"),
                head,
                self._src(joints, "left_shoulder"),
                self._src(joints, "right_shoulder"),
                self._src(joints, "left_elbow"),
                self._src(joints, "right_elbow"),
                self._src(joints, "left_wrist"),
                self._src(joints, "right_wrist"),
                left_hand,
                right_hand,
            ],
            axis=0,
        ).astype(np.float32)

        if out.shape != (24, 3) or not np.isfinite(out).all():
            raise ValueError("Invalid HuMMan SMPL24 joints produced from SAM3 output.")
        return out
