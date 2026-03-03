from __future__ import annotations

from typing import Any

import numpy as np
import rerun.blueprint as rrb
import torch


MODALITY_VIEW_TYPE = {
    "rgb": "2d",
    "depth": "2d",
    "lidar": "3d",
    "mmwave": "3d",
}

MODALITY_VIEW_COUNT_KEY = {
    "rgb": "rgb_cameras_per_sample",
    "depth": "depth_cameras_per_sample",
    "lidar": "lidar_cameras_per_sample",
    "mmwave": "mmwave_cameras_per_sample",
}

MODALITY_CAMERAS_KEY = {
    "rgb": "rgb_cameras",
    "depth": "depth_cameras",
    "lidar": "lidar_cameras",
    "mmwave": "mmwave_cameras",
}


def _resolve_num_views(dataset_params: dict[str, Any], modality: str) -> int:
    count_key = MODALITY_VIEW_COUNT_KEY[modality]
    cameras_key = MODALITY_CAMERAS_KEY[modality]

    has_count = count_key in dataset_params
    has_cameras = cameras_key in dataset_params
    if not has_count and not has_cameras:
        raise ValueError(
            f"Dataset config missing both `{count_key}` and `{cameras_key}` for modality `{modality}`."
        )

    num_views = None
    if has_count:
        num_views = int(dataset_params[count_key])
        if num_views < 1:
            raise ValueError(
                f"Invalid `{count_key}`={num_views} for modality `{modality}`; must be >= 1."
            )
    if has_cameras:
        cameras = dataset_params[cameras_key]
        cameras_count = len(cameras) if isinstance(cameras, (list, tuple)) else int(cameras)
        if cameras_count < 1:
            raise ValueError(
                f"Invalid `{cameras_key}` for modality `{modality}`; must describe at least one camera."
            )
        if num_views is not None and num_views != cameras_count:
            raise ValueError(
                f"Inconsistent view count for modality `{modality}`: "
                f"`{count_key}`={num_views} but `{cameras_key}` has {cameras_count} entries."
            )
        num_views = cameras_count
    return int(num_views)


def build_input_layout_from_config(dataset_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Build modality/view layout from dataset config with strict validation."""
    modality_names = dataset_params.get("modality_names")
    if not modality_names:
        raise ValueError(
            "Dataset config must define non-empty `modality_names` for rerun layout construction."
        )

    layout: list[dict[str, Any]] = []
    for modality in modality_names:
        count_key = MODALITY_VIEW_COUNT_KEY.get(modality)
        if count_key is None:
            raise ValueError(
                f"Unsupported modality `{modality}` in `modality_names`: "
                f"expected one of {sorted(MODALITY_VIEW_COUNT_KEY)}."
            )
        num_views = _resolve_num_views(dataset_params, modality)

        layout.append(
            {
                "modality": modality,
                "num_views": num_views,
                "view_type": MODALITY_VIEW_TYPE.get(modality, "3d"),
            }
        )
    return layout


def create_rerun_blueprint(input_layout: list[dict[str, Any]]) -> rrb.Horizontal:
    """Create a standard rerun layout for inputs + front/side output views."""
    left_views = []
    for item in input_layout:
        modality = item["modality"]
        num_views = int(item["num_views"])
        view_type = item["view_type"]
        for view_idx in range(num_views):
            origin = f"/world/inputs/{modality}/view_{view_idx}"
            name = f"{modality.upper()} View {view_idx}"
            if view_type == "2d":
                left_views.append(rrb.Spatial2DView(name=name, origin=origin))
            else:
                left_views.append(rrb.Spatial3DView(name=name, origin=origin))

    if not left_views:
        raise ValueError("Cannot build rerun blueprint: no input views were configured.")

    return rrb.Horizontal(
        rrb.Vertical(*left_views),
        rrb.Vertical(
            rrb.Spatial3DView(name="Front View", origin="/world/front"),
            rrb.Spatial3DView(name="Side View", origin="/world/side"),
        ),
    )


def split_sample_views(
    data: Any,
    expected_views: int | None = None,
    modality: str | None = None,
) -> list[np.ndarray]:
    """Split modality tensor/array into per-view arrays with optional validation."""
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    if arr.ndim in (2, 3, 4):
        views = [arr]
    elif arr.ndim >= 5:
        views = [arr[i] for i in range(arr.shape[0])]
    else:
        raise ValueError(
            f"Unsupported input rank {arr.ndim} for modality `{modality or 'unknown'}` with shape {arr.shape}."
        )

    if expected_views is not None and len(views) != expected_views:
        raise ValueError(
            f"Configured {expected_views} view(s) for modality `{modality or 'unknown'}` "
            f"but sample contains {len(views)} view(s)."
        )

    return views
