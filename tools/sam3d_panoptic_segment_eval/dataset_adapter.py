from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from misc.registry import create_dataset
from misc.utils import load_cfg, merge_args_cfg


@dataclass(frozen=True)
class DatasetContext:
    cfg_path: str
    split: str
    camera: str
    hparams: Any
    dataset_cfg: dict
    pipeline_cfg: list
    dataset: Any
    denorm_params: dict | None


@dataclass(frozen=True)
class SegmentRecord:
    sequence_name: str
    camera_name: str
    segment_index: int
    segment_length: int
    start_frame_id: int
    end_frame_id: int
    num_frames: int
    sample_indices: tuple[int, ...]
    frame_ids: tuple[int, ...]


class _MockArgs:
    checkpoint_path = ""
    gpus = 1
    num_workers = 0
    batch_size = 1
    batch_size_eva = 1
    pin_memory = False
    prefetch_factor = 2
    use_wandb = False
    save_test_preds = False


def _resolve_dataset_cfg(hparams: Any, split: str) -> tuple[dict, list]:
    if split == "train":
        return hparams.train_dataset, hparams.train_pipeline
    if split == "val":
        return hparams.val_dataset, hparams.val_pipeline
    if split == "test":
        return hparams.test_dataset, hparams.test_pipeline
    raise ValueError(f"Unsupported split={split}. Expected train/val/test.")


def load_dataset_context(cfg_path: str, split: str, camera: str) -> DatasetContext:
    cfg = load_cfg(cfg_path)
    hparams = merge_args_cfg(_MockArgs(), cfg)
    dataset_cfg, pipeline_cfg = _resolve_dataset_cfg(hparams, split)
    dataset_cfg = copy.deepcopy(dataset_cfg)
    params = copy.deepcopy(dataset_cfg["params"])

    modality_names = [str(x).lower() for x in params.get("modality_names", [])]
    if "rgb" not in modality_names:
        raise ValueError(
            f"SAM3 Panoptic segment evaluation requires RGB input, got modality_names={modality_names}."
        )
    if int(params.get("seq_len", 1)) != 1:
        raise ValueError(
            f"SAM3 Panoptic segment evaluation requires dataset seq_len=1, got {params.get('seq_len')}."
        )
    if int(params.get("rgb_cameras_per_sample", 1)) != 1:
        params["rgb_cameras_per_sample"] = 1
    params["rgb_cameras"] = [camera]
    dataset_cfg["params"] = params

    dataset, _ = create_dataset(dataset_cfg["name"], dataset_cfg["params"], pipeline_cfg)
    if not hasattr(dataset, "data_list") or not hasattr(dataset, "sequence_data"):
        raise TypeError(
            f"Dataset `{dataset_cfg['name']}` does not expose `data_list` and `sequence_data`, "
            "which are required for deterministic segment construction."
        )

    has_camera = any(camera in list(info.get("rgb_cams", [])) for info in dataset.sequence_data.values())
    if not has_camera:
        raise ValueError(
            f"Requested camera `{camera}` does not exist in the indexed Panoptic data for split `{split}`."
        )
    if len(dataset) == 0:
        raise ValueError(
            f"Requested camera `{camera}` has no samples in split `{split}` after config filtering."
        )

    return DatasetContext(
        cfg_path=str(Path(cfg_path).expanduser().resolve()),
        split=split,
        camera=camera,
        hparams=hparams,
        dataset_cfg=dataset_cfg,
        pipeline_cfg=pipeline_cfg,
        dataset=dataset,
        denorm_params=getattr(hparams, "vis_denorm_params", None),
    )


def build_segments(
    dataset_ctx: DatasetContext,
    *,
    segment_length: int,
    max_segments: int | None = None,
) -> list[SegmentRecord]:
    if segment_length <= 0:
        raise ValueError(f"segment_length must be > 0, got {segment_length}.")

    dataset = dataset_ctx.dataset
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for sample_idx, data_info in enumerate(dataset.data_list):
        seq_name = str(data_info["seq_name"])
        if dataset_ctx.camera not in list(data_info.get("rgb_cameras", [])):
            continue
        frame_ids = list(dataset.sequence_data[seq_name]["frame_ids"])
        start_frame = int(data_info["start_frame"])
        if start_frame < 0 or start_frame >= len(frame_ids):
            raise ValueError(
                f"Invalid start_frame={start_frame} for seq={seq_name}; frame_ids={len(frame_ids)}."
            )
        body_frame_id = int(frame_ids[start_frame])
        grouped.setdefault((seq_name, dataset_ctx.camera), []).append((sample_idx, body_frame_id))

    if not grouped:
        raise ValueError(
            f"No samples remain for camera `{dataset_ctx.camera}` in split `{dataset_ctx.split}`."
        )

    segments: list[SegmentRecord] = []
    for (seq_name, camera_name), items in sorted(grouped.items()):
        sorted_items = sorted(items, key=lambda pair: pair[1])
        sample_indices = [idx for idx, _ in sorted_items]
        frame_ids = [fid for _, fid in sorted_items]
        num_full_segments = len(sorted_items) // segment_length
        for segment_index in range(num_full_segments):
            start = segment_index * segment_length
            end = start + segment_length
            segment_sample_indices = tuple(sample_indices[start:end])
            segment_frame_ids = tuple(frame_ids[start:end])
            if len(segment_sample_indices) != segment_length:
                raise AssertionError("Tail dropping failed; segment length mismatch encountered.")
            if len({dataset.data_list[idx]["seq_name"] for idx in segment_sample_indices}) != 1:
                raise ValueError(
                    f"Cross-sequence segment boundary violation detected in {seq_name}/{camera_name}."
                )
            segments.append(
                SegmentRecord(
                    sequence_name=seq_name,
                    camera_name=camera_name,
                    segment_index=segment_index,
                    segment_length=segment_length,
                    start_frame_id=int(segment_frame_ids[0]),
                    end_frame_id=int(segment_frame_ids[-1]),
                    num_frames=len(segment_frame_ids),
                    sample_indices=segment_sample_indices,
                    frame_ids=segment_frame_ids,
                )
            )
            if max_segments is not None and len(segments) >= max_segments:
                return segments

    if not segments:
        raise ValueError(
            f"No complete {segment_length}-frame segments could be built for "
            f"camera `{dataset_ctx.camera}` in split `{dataset_ctx.split}`."
        )
    return segments
