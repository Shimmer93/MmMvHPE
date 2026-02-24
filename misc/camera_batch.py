from typing import Optional, Sequence

import torch


def _normalize_gt_camera_shape(
    gt_camera: torch.Tensor,
    batch_size: int,
    pose_encoding_dim: int = 9,
) -> Optional[torch.Tensor]:
    if not isinstance(gt_camera, torch.Tensor):
        return None

    if gt_camera.dim() == 4:
        if gt_camera.shape[-1] != pose_encoding_dim:
            return None
        if gt_camera.shape[0] == batch_size:
            return gt_camera[:, :, -1, :].mean(dim=1)
        if gt_camera.shape[0] == 1 and batch_size > 1:
            gt_camera = gt_camera.expand(batch_size, -1, -1, -1)
            return gt_camera[:, :, -1, :].mean(dim=1)
        if batch_size == 1:
            flat = gt_camera.reshape(-1, gt_camera.shape[-2], gt_camera.shape[-1])
            return flat[:, -1, :].mean(dim=0, keepdim=True)
        return None

    if gt_camera.dim() == 3:
        if gt_camera.shape[-1] != pose_encoding_dim:
            return None
        if gt_camera.shape[0] == batch_size:
            gt_camera = gt_camera[:, -1]
        elif gt_camera.shape[0] == 1:
            gt_camera = gt_camera[:, -1]
            if batch_size > 1:
                gt_camera = gt_camera.expand(batch_size, -1)
        elif batch_size == 1:
            gt_camera = gt_camera[:, -1, :].mean(dim=0, keepdim=True)
        else:
            return None

    if gt_camera.dim() == 2:
        if gt_camera.shape[-1] != pose_encoding_dim:
            return None
        if gt_camera.shape[0] == batch_size:
            return gt_camera
        if gt_camera.shape[0] == 1 and batch_size > 1:
            return gt_camera.expand(batch_size, -1)
        if batch_size == 1:
            return gt_camera[-1:].contiguous()
        return None

    if gt_camera.dim() == 1:
        if gt_camera.shape[0] != pose_encoding_dim:
            return None
        gt_camera = gt_camera.unsqueeze(0)
        if batch_size > 1:
            gt_camera = gt_camera.expand(batch_size, -1)
        return gt_camera

    return None


def prepare_gt_camera_batch(
    gt_camera,
    batch_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    pose_encoding_dim: int = 9,
) -> Optional[torch.Tensor]:
    if gt_camera is None:
        return None

    if isinstance(gt_camera, (list, tuple)):
        out = torch.full(
            (batch_size, pose_encoding_dim),
            float("nan"),
            device=device,
            dtype=dtype,
        )
        for i, value in enumerate(gt_camera[:batch_size]):
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=dtype)
            value = value.to(device=device, dtype=dtype)
            value = _normalize_gt_camera_shape(
                value,
                batch_size=1,
                pose_encoding_dim=pose_encoding_dim,
            )
            if value is not None:
                out[i] = value[0]
        return out

    if not isinstance(gt_camera, torch.Tensor):
        gt_camera = torch.as_tensor(gt_camera, dtype=dtype)
    gt_camera = gt_camera.to(device=device, dtype=dtype)
    return _normalize_gt_camera_shape(
        gt_camera,
        batch_size=batch_size,
        pose_encoding_dim=pose_encoding_dim,
    )


def get_gt_camera_encoding(
    data_batch,
    modality: str,
    batch_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    pose_encoding_dim: int = 9,
) -> Optional[torch.Tensor]:
    gt_camera = data_batch.get(f"gt_camera_{modality}", None)
    return prepare_gt_camera_batch(
        gt_camera=gt_camera,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        pose_encoding_dim=pose_encoding_dim,
    )


def collect_gt_camera_encodings(
    data_batch,
    modalities: Sequence[str],
    batch_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    pose_encoding_dim: int = 9,
) -> Optional[torch.Tensor]:
    if modalities is None:
        return None
    modalities = list(modalities)
    if len(modalities) == 0:
        return None

    out = torch.full(
        (batch_size, len(modalities), pose_encoding_dim),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    has_any = False
    for m_idx, modality in enumerate(modalities):
        gt = get_gt_camera_encoding(
            data_batch=data_batch,
            modality=str(modality).lower(),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            pose_encoding_dim=pose_encoding_dim,
        )
        if gt is None:
            continue
        out[:, m_idx] = gt
        has_any = True
    return out if has_any else None
