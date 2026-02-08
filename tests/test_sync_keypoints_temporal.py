import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.transforms.camera_param_transform import SyncKeypointsWithCameraEncoding
from misc.pose_enc import extri_intri_to_pose_encoding


def _build_identity_camera(batch_size: int, seq_len: int):
    extrinsics = torch.zeros(batch_size, seq_len, 3, 4, dtype=torch.float32)
    intrinsics = torch.zeros(batch_size, seq_len, 3, 3, dtype=torch.float32)
    extrinsics[..., :3, :3] = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    intrinsics[..., 0, 0] = 500.0
    intrinsics[..., 1, 1] = 500.0
    intrinsics[..., 0, 2] = 112.0
    intrinsics[..., 1, 2] = 112.0
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics


def test_project_to_image_broadcasts_non_temporal_keypoints():
    batch_size, seq_len, num_joints = 2, 4, 24
    extrinsics, intrinsics = _build_identity_camera(batch_size, seq_len)

    # Single-frame keypoints should be broadcast to all B and S.
    points = torch.randn(num_joints, 3, dtype=torch.float32)
    points[:, 2] += 2.0  # Keep points in front of camera.

    out = SyncKeypointsWithCameraEncoding._project_to_image(points, extrinsics, intrinsics)
    assert out.shape == (batch_size, seq_len, num_joints, 2)


def test_sync_keypoints_preserves_temporal_length():
    seq_len, num_joints = 5, 24
    extrinsics, intrinsics = _build_identity_camera(1, seq_len)
    gt_camera = extri_intri_to_pose_encoding(
        extrinsics,
        intrinsics,
        image_size_hw=(224, 224),
        pose_encoding_type="absT_quaR_FoV",
    ).squeeze(0)

    sample = {
        "modalities": ["rgb"],
        "input_rgb": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(seq_len)],
        "gt_camera_rgb": gt_camera,
        "gt_keypoints": torch.randn(num_joints, 3, dtype=torch.float32),
    }
    sample["gt_keypoints"][:, 2] += 2.0

    transform = SyncKeypointsWithCameraEncoding(pose_encoding_type="absT_quaR_FoV")
    out = transform(sample)["gt_keypoints_2d_rgb"]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (seq_len, num_joints, 2)
