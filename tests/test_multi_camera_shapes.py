import numpy as np
import pytest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.transforms.camera_param_transform import CameraParamToPoseEncoding, SyncKeypointsWithCameraEncoding
from datasets.transforms.pc_transforms import PCPad
from metrics.camera import CameraPoseAUC, CameraRotationAngleError, CameraTranslationL2Error
from misc.pose_enc import extri_intri_to_pose_encoding
from models.heads.keypoint_camera_gcn_head_v5 import KeypointCameraGCNHeadV5


def _camera_dict(tx: float = 0.0):
    intrinsic = np.array(
        [[500.0, 0.0, 112.0], [0.0, 500.0, 112.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsic = np.hstack(
        [np.eye(3, dtype=np.float32), np.array([[tx], [0.0], [0.0]], dtype=np.float32)]
    )
    return {"intrinsic": intrinsic, "extrinsic": extrinsic}


def _rgb_frames(seq_len: int):
    return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(seq_len)]


def test_camera_param_to_pose_encoding_supports_multi_view_and_single_view():
    seq_len = 3
    transform = CameraParamToPoseEncoding(pose_encoding_type="absT_quaR_FoV")

    single = {
        "modalities": ["rgb"],
        "input_rgb": _rgb_frames(seq_len),
        "rgb_camera": _camera_dict(0.0),
    }
    single_out = transform(single)
    assert isinstance(single_out["gt_camera_rgb"], torch.Tensor)
    assert single_out["gt_camera_rgb"].shape == (seq_len, 9)

    multi = {
        "modalities": ["rgb"],
        "input_rgb": [_rgb_frames(seq_len), _rgb_frames(seq_len)],
        "rgb_camera": [_camera_dict(0.0), _camera_dict(0.2)],
    }
    multi_out = transform(multi)
    assert isinstance(multi_out["gt_camera_rgb"], torch.Tensor)
    assert multi_out["gt_camera_rgb"].shape == (2, seq_len, 9)


def test_sync_keypoints_with_camera_encoding_supports_multi_view():
    views, seq_len, num_joints = 2, 4, 24
    extrinsics = torch.zeros(views, seq_len, 3, 4, dtype=torch.float32)
    intrinsics = torch.zeros(views, seq_len, 3, 3, dtype=torch.float32)
    extrinsics[..., :3, :3] = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    intrinsics[..., 0, 0] = 500.0
    intrinsics[..., 1, 1] = 500.0
    intrinsics[..., 0, 2] = 112.0
    intrinsics[..., 1, 2] = 112.0
    intrinsics[..., 2, 2] = 1.0

    gt_camera = extri_intri_to_pose_encoding(
        extrinsics,
        intrinsics,
        image_size_hw=(224, 224),
        pose_encoding_type="absT_quaR_FoV",
    )

    sample = {
        "modalities": ["rgb"],
        "input_rgb": [_rgb_frames(seq_len), _rgb_frames(seq_len)],
        "gt_camera_rgb": gt_camera,
        "gt_keypoints": torch.randn(num_joints, 3, dtype=torch.float32),
    }
    sample["gt_keypoints"][:, 2] += 2.0

    transform = SyncKeypointsWithCameraEncoding(pose_encoding_type="absT_quaR_FoV")
    out = transform(sample)["gt_keypoints_2d_rgb"]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (views, seq_len, num_joints, 2)


def test_pc_pad_supports_multi_view_sequences():
    sample = {
        "input_lidar": [
            [np.random.randn(5, 4).astype(np.float32), np.random.randn(12, 4).astype(np.float32)],
            [np.random.randn(2, 4).astype(np.float32), np.random.randn(20, 4).astype(np.float32)],
        ]
    }

    out = PCPad(num_points=16, pad_mode="repeat", keys=["input_lidar"])(sample)
    assert len(out["input_lidar"]) == 2
    for view_seq in out["input_lidar"]:
        for pc in view_seq:
            assert pc.shape == (16, 4)


def test_keypoint_camera_head_normalizes_multiview_shapes():
    head = KeypointCameraGCNHeadV5(losses=[])
    batch_size, views, seq_len, num_joints = 2, 3, 4, 24

    gt_camera = torch.randn(batch_size, views, seq_len, 9, dtype=torch.float32)
    normalized = head._normalize_gt_camera_shape(gt_camera, batch_size=batch_size)
    expected = gt_camera[:, :, -1, :].mean(dim=1)
    assert normalized.shape == (batch_size, 9)
    assert torch.allclose(normalized, expected)

    keypoints = torch.randn(batch_size, views, seq_len, num_joints, 3, dtype=torch.float32)
    keypoints_bstjc = head._ensure_bstjc(keypoints, batch_size=batch_size)
    assert keypoints_bstjc.shape == (batch_size, seq_len, num_joints, 3)
    assert torch.allclose(keypoints_bstjc, keypoints.mean(dim=1))


def test_camera_metrics_support_multiview_gt_camera():
    batch_size, num_modalities, views, seq_len = 3, 2, 2, 4
    gt_camera = torch.zeros(batch_size, views, seq_len, 9, dtype=torch.float32)
    gt_camera[..., :3] = torch.randn(batch_size, views, seq_len, 3)
    gt_camera[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    gt_camera[..., 7:9] = torch.tensor([1.0, 1.0], dtype=torch.float32)

    gt_last_mean = gt_camera[:, :, -1, :].mean(dim=1)
    pred_cameras = torch.zeros(batch_size, num_modalities, 9, dtype=torch.float32)
    pred_cameras[:, 0, :] = gt_last_mean

    preds = {"pred_cameras": pred_cameras}
    targets = {
        "gt_camera_rgb": gt_camera,
        "modalities": [["rgb", "lidar"]],
    }

    trans = CameraTranslationL2Error(modality="rgb")(preds, targets)
    rot = CameraRotationAngleError(modality="rgb")(preds, targets)
    auc = CameraPoseAUC(modality="rgb", max_threshold=30)(preds, targets)

    assert trans == pytest.approx(0.0, abs=1e-7)
    assert rot == pytest.approx(0.0, abs=1e-7)
    assert auc == pytest.approx(1.0, abs=1e-7)
