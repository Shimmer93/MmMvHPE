from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "run"


def make_run_dir(
    *,
    cfg_path: str,
    split: str,
    camera: str,
    segment_length: int,
    output_root: str = "logs/sam3d_panoptic_segment_eval",
    run_name: str | None = None,
    segmentor_name: str = "none",
    use_mask: bool = False,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        stem = Path(cfg_path).stem
        mask_tag = "mask" if use_mask else "nomask"
        run_name = f"{stem}__{split}__{camera}__seg{segment_length}__{segmentor_name}__{mask_tag}"
    run_dir = root / _slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def dump_json(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(make_json_serializable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
