import numpy as np
import pytest
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import PCMPJPE, SMPL_PCMPJPE
from misc.registry import create_metric


def _rot_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _apply_rigid(points, rot, transl):
    return np.einsum("ij,bkj->bki", rot, points) + transl.reshape(1, 1, 3)


def _build_non_symmetric_skeleton(batch=2, joints=24):
    base = np.linspace(0.0, 1.0, joints * 3, dtype=np.float32).reshape(joints, 3)
    base[0] = np.array([0.10, 0.05, 0.00], dtype=np.float32)  # neck
    base[2] = np.array([0.05, -0.02, 0.03], dtype=np.float32)  # bodycenter
    base[6] = np.array([-0.20, -0.30, 0.05], dtype=np.float32)  # lhip
    base[12] = np.array([0.25, -0.28, -0.02], dtype=np.float32)  # rhip
    return np.stack([base + (i * 0.01) for i in range(batch)], axis=0).astype(np.float32)


def test_pcmpjpe_is_near_zero_for_global_rigid_difference():
    gt = _build_non_symmetric_skeleton(batch=3)
    rot = _rot_z(theta=0.7)
    transl = np.array([0.4, -0.2, 0.15], dtype=np.float32)
    pred = _apply_rigid(gt, rot, transl)

    metric = PCMPJPE(
        use_smpl=False,
        pelvis_idx=0,
        neck_idx=0,
        bodycenter_idx=2,
        lhip_idx=6,
        rhip_idx=12,
    )
    value = metric(
        preds={"pred_keypoints": torch.from_numpy(pred)},
        targets={"gt_keypoints": torch.from_numpy(gt)},
    )
    assert value == pytest.approx(0.0, abs=1e-5)


def test_pcmpjpe_raises_when_required_orientation_keypoints_missing():
    gt = np.zeros((2, 10, 3), dtype=np.float32)
    pred = gt.copy()
    metric = PCMPJPE(
        use_smpl=False,
        pelvis_idx=0,
        neck_idx=0,
        bodycenter_idx=2,
        lhip_idx=6,
        rhip_idx=12,
    )
    with pytest.raises(ValueError, match="Need keypoints indices"):
        metric(
            preds={"pred_keypoints": torch.from_numpy(pred)},
            targets={"gt_keypoints": torch.from_numpy(gt)},
        )


def test_smpl_pcmpjpe_is_near_zero_for_global_rigid_difference():
    gt = _build_non_symmetric_skeleton(batch=2)
    angle = 0.45
    rot = _rot_z(theta=angle)
    transl = np.array([-0.1, 0.25, 0.08], dtype=np.float32)
    pred = _apply_rigid(gt, rot, transl)

    pred_global_orient = np.array([[0.0, 0.0, angle], [0.0, 0.0, angle]], dtype=np.float32)
    gt_global_orient = np.zeros((2, 3), dtype=np.float32)

    metric = SMPL_PCMPJPE(pelvis_idx=0)
    value = metric(
        preds={
            "pred_smpl_keypoints": torch.from_numpy(pred),
            "pred_smpl": {"global_orient": torch.from_numpy(pred_global_orient)},
        },
        targets={
            "gt_keypoints": torch.from_numpy(gt),
            "gt_smpl": {"global_orient": torch.from_numpy(gt_global_orient)},
        },
    )
    assert value == pytest.approx(0.0, abs=1e-5)


def test_smpl_pcmpjpe_raises_when_root_orientation_missing():
    gt = _build_non_symmetric_skeleton(batch=1)
    pred = gt.copy()
    metric = SMPL_PCMPJPE(pelvis_idx=0)
    with pytest.raises(ValueError, match="Missing gt_smpl_params"):
        metric(
            preds={
                "pred_smpl_keypoints": torch.from_numpy(pred),
                "pred_smpl_params": torch.zeros((1, 82), dtype=torch.float32),
            },
            targets={"gt_keypoints": torch.from_numpy(gt)},
        )


def test_metric_registry_resolves_new_metric_names():
    pcmpjpe = create_metric(
        "PCMPJPE",
        {
            "use_smpl": False,
            "pelvis_idx": 0,
            "neck_idx": 0,
            "bodycenter_idx": 2,
            "lhip_idx": 6,
            "rhip_idx": 12,
        },
    )
    smpl_pcmpjpe = create_metric("SMPL_PCMPJPE", {"pelvis_idx": 0})
    assert isinstance(pcmpjpe, PCMPJPE)
    assert isinstance(smpl_pcmpjpe, SMPL_PCMPJPE)
