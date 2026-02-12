import os
import time
import json
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target, dim=-1).mean()


def _format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
    log_interval: int = 20,
    epoch_idx: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    start_time = time.time()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_samples = 0

    num_steps = len(dataloader)
    for step_idx, batch in enumerate(dataloader, start=1):
        input_lidar = batch["input_lidar"].to(device, non_blocking=True).float()
        gt_keypoints = batch["gt_keypoints"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        pred_keypoints = model(input_lidar)
        loss = F.mse_loss(pred_keypoints, gt_keypoints)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = input_lidar.shape[0]
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_mpjpe += float(mpjpe(pred_keypoints.detach(), gt_keypoints).item()) * batch_size

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == num_steps):
            avg_loss = total_loss / max(total_samples, 1)
            avg_mpjpe = total_mpjpe / max(total_samples, 1)
            elapsed = _format_seconds(time.time() - start_time)
            if epoch_idx is not None and num_epochs is not None:
                print(
                    f"[Train] Epoch {epoch_idx + 1}/{num_epochs} "
                    f"Step {step_idx}/{num_steps} "
                    f"loss={avg_loss:.6f} mpjpe={avg_mpjpe:.6f} elapsed={elapsed}"
                )
            else:
                print(
                    f"[Train] Step {step_idx}/{num_steps} "
                    f"loss={avg_loss:.6f} mpjpe={avg_mpjpe:.6f} elapsed={elapsed}"
                )

    return {
        "loss": total_loss / max(total_samples, 1),
        "mpjpe": total_mpjpe / max(total_samples, 1),
        "elapsed_sec": time.time() - start_time,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    log_interval: int = 20,
    epoch_idx: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    start_time = time.time()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_samples = 0

    num_steps = len(dataloader)
    for step_idx, batch in enumerate(dataloader, start=1):
        input_lidar = batch["input_lidar"].to(device, non_blocking=True).float()
        gt_keypoints = batch["gt_keypoints"].to(device, non_blocking=True).float()

        pred_keypoints = model(input_lidar)
        loss = F.mse_loss(pred_keypoints, gt_keypoints)

        batch_size = input_lidar.shape[0]
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_mpjpe += float(mpjpe(pred_keypoints, gt_keypoints).item()) * batch_size

        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == num_steps):
            avg_loss = total_loss / max(total_samples, 1)
            avg_mpjpe = total_mpjpe / max(total_samples, 1)
            elapsed = _format_seconds(time.time() - start_time)
            if epoch_idx is not None and num_epochs is not None:
                print(
                    f"[Val]   Epoch {epoch_idx + 1}/{num_epochs} "
                    f"Step {step_idx}/{num_steps} "
                    f"loss={avg_loss:.6f} mpjpe={avg_mpjpe:.6f} elapsed={elapsed}"
                )
            else:
                print(
                    f"[Val]   Step {step_idx}/{num_steps} "
                    f"loss={avg_loss:.6f} mpjpe={avg_mpjpe:.6f} elapsed={elapsed}"
                )

    return {
        "loss": total_loss / max(total_samples, 1),
        "mpjpe": total_mpjpe / max(total_samples, 1),
        "elapsed_sec": time.time() - start_time,
    }


@torch.no_grad()
def test_and_save_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    log_interval: int = 20,
) -> Dict[str, float]:
    model.eval()
    start_time = time.time()
    all_preds = []
    all_targets = []
    all_sample_ids = []
    total_mpjpe = 0.0
    total_samples = 0

    num_steps = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        input_lidar = batch["input_lidar"].to(device, non_blocking=True).float()
        gt_keypoints = batch["gt_keypoints"].to(device, non_blocking=True).float()
        pred_keypoints = model(input_lidar)

        batch_size = input_lidar.shape[0]
        total_samples += batch_size
        total_mpjpe += float(mpjpe(pred_keypoints, gt_keypoints).item()) * batch_size

        all_preds.append(pred_keypoints.cpu().numpy())
        all_targets.append(gt_keypoints.cpu().numpy())

        sample_ids = batch.get("sample_id", None)
        if sample_ids is None:
            all_sample_ids.extend([f"sample_{batch_idx}_{i}" for i in range(batch_size)])
        else:
            all_sample_ids.extend(list(sample_ids))

        step_idx = batch_idx + 1
        if log_interval > 0 and (step_idx % log_interval == 0 or step_idx == num_steps):
            avg_mpjpe = total_mpjpe / max(total_samples, 1)
            elapsed = _format_seconds(time.time() - start_time)
            print(
                f"[Test]  Step {step_idx}/{num_steps} "
                f"mpjpe={avg_mpjpe:.6f} elapsed={elapsed}"
            )

    pred_arr = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0, 0, 3), dtype=np.float32)
    target_arr = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 0, 3), dtype=np.float32)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        save_path,
        sample_id=np.asarray(all_sample_ids),
        pred_keypoints=pred_arr,
        gt_keypoints=target_arr,
    )

    return {
        "mpjpe": total_mpjpe / max(total_samples, 1),
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
) -> Dict[str, float]:
    model.eval()
    start_time = time.time()
    predictions = []

    num_steps = len(dataloader)
    for step_idx, batch in enumerate(dataloader, start=1):
        input_lidar = batch["input_lidar"].to(device, non_blocking=True).float()
        pred_keypoints = model(input_lidar).cpu().numpy()

        frame_paths = batch.get("frame_path", None)
        if frame_paths is None:
            sample_ids = batch.get("sample_id", [f"sample_{step_idx}_{i}" for i in range(input_lidar.shape[0])])
            frame_paths = [str(x) for x in sample_ids]

        for i in range(pred_keypoints.shape[0]):
            kp = pred_keypoints[i].astype(np.float32)
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
            print(
                f"[Export] Step {step_idx}/{num_steps} "
                f"frames={len(predictions)} elapsed={elapsed}"
            )

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
