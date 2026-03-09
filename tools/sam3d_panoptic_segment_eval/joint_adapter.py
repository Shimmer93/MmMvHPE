from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info


PANOPTIC_DATASET19_NAMES = [
    "neck",
    "nose",
    "mid_hip",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_eye",
    "left_ear",
    "right_eye",
    "right_ear",
]

PANOPTIC_DATASET19_BONES = [
    [0, 1],
    [0, 3],
    [3, 4],
    [4, 5],
    [0, 2],
    [2, 6],
    [6, 7],
    [7, 8],
    [2, 12],
    [12, 13],
    [13, 14],
    [0, 9],
    [9, 10],
    [10, 11],
    [1, 15],
    [15, 16],
    [1, 17],
    [17, 18],
]

DIRECT_MHR70_TO_PANOPTIC = {
    "neck": "neck",
    "nose": "nose",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",
    "left_eye": "left_eye",
    "left_ear": "left_ear",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",
    "right_eye": "right_eye",
    "right_ear": "right_ear",
}


@dataclass(frozen=True)
class JointMappingRow:
    panoptic_joint: str
    source_joint: str
    derived: bool


class SAM3ToPanopticCOCO19Adapter:
    def __init__(self) -> None:
        keypoint_info = mhr70_pose_info["keypoint_info"]
        self.source_name_to_idx = {
            str(meta["name"]).strip(): int(kpt_id)
            for kpt_id, meta in keypoint_info.items()
        }
        missing = sorted(
            source_name
            for source_name in DIRECT_MHR70_TO_PANOPTIC.values()
            if source_name not in self.source_name_to_idx
        )
        if missing:
            raise ValueError(
                "SAM3 MHR70 metadata is missing required joints for Panoptic dataset 19-joint conversion: "
                + ", ".join(missing)
            )
        for needed in ("left_hip", "right_hip"):
            if needed not in self.source_name_to_idx:
                raise ValueError(
                    f"SAM3 MHR70 metadata is missing `{needed}`, which is required to derive `mid_hip`."
                )

    def mapping_rows(self) -> list[JointMappingRow]:
        rows: list[JointMappingRow] = []
        for panoptic_name in PANOPTIC_DATASET19_NAMES:
            if panoptic_name == "mid_hip":
                rows.append(
                    JointMappingRow(
                        panoptic_joint=panoptic_name,
                        source_joint="0.5 * (left_hip + right_hip)",
                        derived=True,
                    )
                )
            else:
                source_name = DIRECT_MHR70_TO_PANOPTIC[panoptic_name]
                rows.append(
                    JointMappingRow(
                        panoptic_joint=panoptic_name,
                        source_joint=source_name,
                        derived=False,
                    )
                )
        return rows

    def adapt(self, sam_keypoints_3d: np.ndarray) -> np.ndarray:
        joints = np.asarray(sam_keypoints_3d, dtype=np.float32)
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError(f"SAM keypoints must have shape (J,3), got {joints.shape}.")

        target = np.zeros((len(PANOPTIC_DATASET19_NAMES), 3), dtype=np.float32)
        for target_idx, panoptic_name in enumerate(PANOPTIC_DATASET19_NAMES):
            if panoptic_name == "mid_hip":
                left_hip = joints[self.source_name_to_idx["left_hip"]]
                right_hip = joints[self.source_name_to_idx["right_hip"]]
                target[target_idx] = 0.5 * (left_hip + right_hip)
                continue
            source_name = DIRECT_MHR70_TO_PANOPTIC.get(panoptic_name)
            if source_name is None:
                raise ValueError(f"No SAM3 mapping defined for Panoptic joint `{panoptic_name}`.")
            source_idx = self.source_name_to_idx.get(source_name)
            if source_idx is None:
                raise ValueError(
                    f"SAM3 MHR70 metadata does not expose source joint `{source_name}` "
                    f"needed for Panoptic joint `{panoptic_name}`."
                )
            target[target_idx] = joints[source_idx]

        if not np.isfinite(target).all():
            raise ValueError("Non-finite values encountered while converting SAM3 joints to Panoptic dataset order.")
        return target
