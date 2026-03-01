import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

PC_CENTERED_TARGET_KEY = "gt_keypoints_pc_centered_input_lidar"
PC_AFFINE_KEY = "input_lidar_affine"
GLOBAL_TARGET_KEY = "gt_keypoints"


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"mpjpe shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}.")
    return torch.norm(pred - target, dim=-1).mean()


def _format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_progress(msg: str, *, final: bool, overwrite: bool) -> None:
    if overwrite and sys.stdout.isatty():
        print(msg.ljust(180), end="\n" if final else "\r", flush=True)
        return
    print(msg, flush=True)


def _to_float_tensor(x, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32) if device is not None else x.float()
    t = torch.as_tensor(x, dtype=torch.float32)
    return t.to(device=device) if device is not None else t


def _get_required_tensor(batch: Dict[str, object], key: str) -> torch.Tensor:
    value = batch.get(key, None)
    if value is None:
        raise KeyError(f"Missing required key `{key}` in batch.")

    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"`{key}` list is empty.")
        items = []
        for i, item in enumerate(value):
            if item is None:
                raise ValueError(f"`{key}` contains None at batch index {i}.")
            items.append(_to_float_tensor(item))
        return torch.stack(items, dim=0)
    raise TypeError(f"Unsupported `{key}` type: {type(value).__name__}.")


def _get_optional_tensor(batch: Dict[str, object], key: str) -> Optional[torch.Tensor]:
    value = batch.get(key, None)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if any(v is None for v in value):
            return None
        return torch.stack([_to_float_tensor(v) for v in value], dim=0)
    return None


def _get_lidar_centers(batch: Dict[str, object]) -> torch.Tensor:
    affine = _get_required_tensor(batch, PC_AFFINE_KEY)
    if affine.dim() == 2:
        affine = affine.unsqueeze(0)
    if affine.dim() != 3 or affine.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected `{PC_AFFINE_KEY}` as [B,4,4] (or [4,4]), got {tuple(affine.shape)}."
        )
    return -affine[:, :3, 3]


def _restore_centered_keypoints(pred_centered: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    if pred_centered.dim() != 3 or pred_centered.shape[-1] != 3:
        raise ValueError(
            f"Expected centered predictions as [B,J,3], got {tuple(pred_centered.shape)}."
        )
    if centers.dim() != 2 or centers.shape[-1] != 3:
        raise ValueError(f"Expected centers as [B,3], got {tuple(centers.shape)}.")
    if pred_centered.shape[0] != centers.shape[0]:
        raise ValueError(
            f"Batch mismatch between predictions and centers: {pred_centered.shape[0]} vs {centers.shape[0]}."
        )
    return pred_centered + centers.unsqueeze(1)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
    log_interval: int = 20,
    log_overwrite: bool = True,
    epoch_idx: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    start_time = time.time()
    total_loss = 0.0
    total_mpjpe_centered = 0.0
    total_mpjpe_restored = 0.0
    total_samples = 0
    total_restored_samples = 0

    num_steps = len(dataloader)
    for step_idx, batch in enumerate(dataloader, start=1):
        input_lidar = _get_required_tensor(batch, "input_lidar").to(device, non_blocking=True)
        gt_centered = _get_required_tensor(batch, PC_CENTERED_TARGET_KEY).to(device, non_blocking=True)
        lidar_centers = _get_lidar_centers(batch).to(device, non_blocking=True)
        gt_global = _get_optional_tensor(batch, GLOBAL_TARGET_KEY)
        if gt_global is not None:
            gt_global = gt_global.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred_centered = model(input_lidar)
        pred_restored = _restore_centered_keypoints(pred_centered, lidar_centers)

        loss = F.mse_loss(pred_centered, gt_centered)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = input_lidar.shape[0]
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_mpjpe_centered += float(mpjpe(pred_centered.detach(), gt_centered).item()) * batch_size

        if gt_global is not None:
            total_mpjpe_restored += float(mpjpe(pred_restored.detach(), gt_global).item()) * batch_size
            total_restored_samples += batch_size

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == num_steps):
            avg_loss = total_loss / max(total_samples, 1)
            avg_mpjpe_centered = total_mpjpe_centered / max(total_samples, 1)
            avg_mpjpe_restored = (
                total_mpjpe_restored / total_restored_samples
                if total_restored_samples > 0
                else float("nan")
            )
            elapsed = _format_seconds(time.time() - start_time)
            if epoch_idx is not None and num_epochs is not None:
                msg = (
                    f"[Train] Epoch {epoch_idx + 1}/{num_epochs} "
                    f"Step {step_idx}/{num_steps} "
                    f"loss={avg_loss:.6f} "
                    f"mpjpe_centered={avg_mpjpe_centered:.6f} "
                    f"mpjpe_restored={avg_mpjpe_restored:.6f} "
                    f"elapsed={elapsed}"
                )
            else:
                msg = (
                    f"[Train] Step {step_idx}/{num_steps} "
                    f"loss={avg_loss:.6f} "
                    f"mpjpe_centered={avg_mpjpe_centered:.6f} "
                    f"mpjpe_restored={avg_mpjpe_restored:.6f} "
                    f"elapsed={elapsed}"
                )
            _print_progress(msg, final=(step_idx == num_steps), overwrite=log_overwrite)

    return {
        "loss": total_loss / max(total_samples, 1),
        "mpjpe_centered": total_mpjpe_centered / max(total_samples, 1),
        "mpjpe_restored": (
            total_mpjpe_restored / total_restored_samples if total_restored_samples > 0 else float("nan")
        ),
        "elapsed_sec": time.time() - start_time,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    log_interval: int = 20,
    log_overwrite: bool = True,
    epoch_idx: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    start_time = time.time()
    total_loss = 0.0
    total_mpjpe_centered = 0.0
    total_mpjpe_restored = 0.0
    total_samples = 0
    total_restored_samples = 0

    num_steps = len(dataloader)
    for step_idx, batch in enumerate(dataloader, start=1):
        input_lidar = _get_required_tensor(batch, "input_lidar").to(device, non_blocking=True)
        gt_centered = _get_required_tensor(batch, PC_CENTERED_TARGET_KEY).to(device, non_blocking=True)
        lidar_centers = _get_lidar_centers(batch).to(device, non_blocking=True)
        gt_global = _get_optional_tensor(batch, GLOBAL_TARGET_KEY)
        if gt_global is not None:
            gt_global = gt_global.to(device, non_blocking=True)

        pred_centered = model(input_lidar)
        pred_restored = _restore_centered_keypoints(pred_centered, lidar_centers)
        loss = F.mse_loss(pred_centered, gt_centered)

        batch_size = input_lidar.shape[0]
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_mpjpe_centered += float(mpjpe(pred_centered, gt_centered).item()) * batch_size
        if gt_global is not None:
            total_mpjpe_restored += float(mpjpe(pred_restored, gt_global).item()) * batch_size
            total_restored_samples += batch_size

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == num_steps):
            avg_loss = total_loss / max(total_samples, 1)
            avg_mpjpe_centered = total_mpjpe_centered / max(total_samples, 1)
            avg_mpjpe_restored = (
                total_mpjpe_restored / total_restored_samples
                if total_restored_samples > 0
                else float("nan")
            )
            elapsed = _format_seconds(time.time() - start_time)
            if epoch_idx is not None and num_epochs is not None:
                msg = (
                    f"[Val]   Epoch {epoch_idx + 1}/{num_epochs} "
                    f"Step {step_idx}/{num_steps} "
                    f"loss={avg_loss:.6f} "
                    f"mpjpe_centered={avg_mpjpe_centered:.6f} "
                    f"mpjpe_restored={avg_mpjpe_restored:.6f} "
                    f"elapsed={elapsed}"
                )
            else:
                msg = (
                    f"[Val]   Step {step_idx}/{num_steps} "
                    f"loss={avg_loss:.6f} "
                    f"mpjpe_centered={avg_mpjpe_centered:.6f} "
                    f"mpjpe_restored={avg_mpjpe_restored:.6f} "
                    f"elapsed={elapsed}"
                )
            _print_progress(msg, final=(step_idx == num_steps), overwrite=log_overwrite)

    return {
        "loss": total_loss / max(total_samples, 1),
        "mpjpe_centered": total_mpjpe_centered / max(total_samples, 1),
        "mpjpe_restored": (
            total_mpjpe_restored / total_restored_samples if total_restored_samples > 0 else float("nan")
        ),
        "elapsed_sec": time.time() - start_time,
    }


@torch.no_grad()
def test_and_save_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    log_interval: int = 20,
    log_overwrite: bool = True,
) -> Dict[str, float]:
    model.eval()
    start_time = time.time()
    all_pred_centered = []
    all_pred_restored = []
    all_target_centered = []
    all_target_global = []
    all_lidar_centers = []
    all_sample_ids = []
    total_mpjpe_centered = 0.0
    total_mpjpe_restored = 0.0
    total_samples = 0
    total_restored_samples = 0

    num_steps = len(dataloader)
    for batch_idx, batch in enumerate(dataloader, start=1):
        input_lidar = _get_required_tensor(batch, "input_lidar").to(device, non_blocking=True)
        gt_centered = _get_required_tensor(batch, PC_CENTERED_TARGET_KEY).to(device, non_blocking=True)
        lidar_centers = _get_lidar_centers(batch).to(device, non_blocking=True)
        gt_global = _get_optional_tensor(batch, GLOBAL_TARGET_KEY)
        if gt_global is not None:
            gt_global = gt_global.to(device, non_blocking=True)

        pred_centered = model(input_lidar)
        pred_restored = _restore_centered_keypoints(pred_centered, lidar_centers)

        batch_size = input_lidar.shape[0]
        total_samples += batch_size
        total_mpjpe_centered += float(mpjpe(pred_centered, gt_centered).item()) * batch_size
        if gt_global is not None:
            total_mpjpe_restored += float(mpjpe(pred_restored, gt_global).item()) * batch_size
            total_restored_samples += batch_size

        all_pred_centered.append(pred_centered.cpu().numpy())
        all_pred_restored.append(pred_restored.cpu().numpy())
        all_target_centered.append(gt_centered.cpu().numpy())
        if gt_global is None:
            all_target_global.append(np.full(pred_restored.shape, np.nan, dtype=np.float32))
        else:
            all_target_global.append(gt_global.cpu().numpy())
        all_lidar_centers.append(lidar_centers.cpu().numpy())

        sample_ids = batch.get("sample_id", None)
        if sample_ids is None:
            all_sample_ids.extend([f"sample_{batch_idx}_{i}" for i in range(batch_size)])
        else:
            all_sample_ids.extend(list(sample_ids))

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_steps):
            avg_mpjpe_centered = total_mpjpe_centered / max(total_samples, 1)
            avg_mpjpe_restored = (
                total_mpjpe_restored / total_restored_samples
                if total_restored_samples > 0
                else float("nan")
            )
            elapsed = _format_seconds(time.time() - start_time)
            msg = (
                f"[Test]  Step {batch_idx}/{num_steps} "
                f"mpjpe_centered={avg_mpjpe_centered:.6f} "
                f"mpjpe_restored={avg_mpjpe_restored:.6f} "
                f"elapsed={elapsed}"
            )
            _print_progress(msg, final=(batch_idx == num_steps), overwrite=log_overwrite)

    pred_centered_arr = (
        np.concatenate(all_pred_centered, axis=0) if all_pred_centered else np.empty((0, 0, 3), dtype=np.float32)
    )
    pred_restored_arr = (
        np.concatenate(all_pred_restored, axis=0) if all_pred_restored else np.empty((0, 0, 3), dtype=np.float32)
    )
    target_centered_arr = (
        np.concatenate(all_target_centered, axis=0) if all_target_centered else np.empty((0, 0, 3), dtype=np.float32)
    )
    target_global_arr = (
        np.concatenate(all_target_global, axis=0) if all_target_global else np.empty((0, 0, 3), dtype=np.float32)
    )
    lidar_center_arr = (
        np.concatenate(all_lidar_centers, axis=0) if all_lidar_centers else np.empty((0, 3), dtype=np.float32)
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        save_path,
        sample_id=np.asarray(all_sample_ids),
        pred_keypoints=pred_restored_arr,
        pred_keypoints_centered=pred_centered_arr,
        gt_keypoints=target_global_arr,
        gt_keypoints_centered=target_centered_arr,
        input_lidar_center=lidar_center_arr,
    )

    return {
        "mpjpe_centered": total_mpjpe_centered / max(total_samples, 1),
        "mpjpe_restored": (
            total_mpjpe_restored / total_restored_samples if total_restored_samples > 0 else float("nan")
        ),
        "num_samples": float(total_samples),
        "elapsed_sec": time.time() - start_time,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_mpjpe: float,
) -> None:
    state = {
        "epoch": int(epoch),
        "best_val_mpjpe": float(best_val_mpjpe),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str = "cpu",
) -> Tuple[int, float]:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"], strict=True)

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    epoch = int(state.get("epoch", -1))
    best_val_mpjpe = float(state.get("best_val_mpjpe", np.inf))
    return epoch, best_val_mpjpe


@torch.no_grad()
def export_predictions_as_mmpose_json(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    input_dir: str,
    config: str,
    checkpoint: str,
    device_str: str,
    log_interval: int = 20,
    log_overwrite: bool = True,
) -> Dict[str, float]:
    model.eval()
    start_time = time.time()
    predictions = []

    num_steps = len(dataloader)
    for step_idx, batch in enumerate(dataloader, start=1):
        input_lidar = _get_required_tensor(batch, "input_lidar").to(device, non_blocking=True)
        lidar_centers = _get_lidar_centers(batch).to(device, non_blocking=True)
        pred_centered = model(input_lidar)
        pred_restored = _restore_centered_keypoints(pred_centered, lidar_centers).cpu().numpy()

        frame_paths = batch.get("frame_path", None)
        if frame_paths is None:
            sample_ids = batch.get("sample_id", [f"sample_{step_idx}_{i}" for i in range(input_lidar.shape[0])])
            frame_paths = [str(x) for x in sample_ids]

        for i in range(pred_restored.shape[0]):
            kp = pred_restored[i].astype(np.float32)
            kps = np.ones((kp.shape[0],), dtype=np.float32)
            predictions.append(
                {
                    "image_path": str(frame_paths[i]),
                    "instances": [
                        {
                            "keypoints": kp.tolist(),
                            "keypoint_scores": kps.tolist(),
                        }
                    ],
                }
            )

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == num_steps):
            elapsed = _format_seconds(time.time() - start_time)
            msg = (
                f"[Export] Step {step_idx}/{num_steps} "
                f"frames={len(predictions)} elapsed={elapsed}"
            )
            _print_progress(msg, final=(step_idx == num_steps), overwrite=log_overwrite)

    payload = {
        "input_dir": str(input_dir),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "device": str(device_str),
        "num_images": len(predictions),
        "predictions": predictions,
    }
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "num_images": float(len(predictions)),
        "elapsed_sec": time.time() - start_time,
    }
