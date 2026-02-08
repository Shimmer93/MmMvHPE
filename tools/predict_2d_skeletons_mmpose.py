import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
MMPPOSE_ROOT = REPO_ROOT / "third_party" / "mmpose"
if MMPPOSE_ROOT.exists():
    sys.path.insert(0, str(MMPPOSE_ROOT))

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def import_mmpose_apis() -> Tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import merge_data_samples
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("mmdet"):
            raise RuntimeError(
                "Failed to import mmpose because `mmdet` is missing in this environment. "
                "Please install MMDetection (mmdet) first."
            ) from exc
        raise
    return inference_topdown, init_model, merge_data_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict 2D skeletons for all images in a directory with MMPose."
    )
    parser.add_argument("input_dir", help="Input image directory.")
    parser.add_argument("config", help="MMPose config file.")
    parser.add_argument("checkpoint", help="MMPose checkpoint file.")
    parser.add_argument("output", help="Output JSON file path.")
    parser.add_argument(
        "--device", default="cuda:0", help="Device used for inference."
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Image extensions to include.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only process files directly under input_dir.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Raise immediately when an image fails during inference.",
    )
    return parser.parse_args()


@contextlib.contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def is_url(path_or_url: str) -> bool:
    parsed = urlparse(path_or_url)
    return parsed.scheme in {"http", "https"}


def list_images(input_dir: Path, recursive: bool, extensions: List[str]) -> List[Path]:
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    pattern = "*" if not recursive else "**/*"
    image_paths = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in ext_set
    ]
    image_paths.sort(key=lambda p: str(p))
    return image_paths


def to_serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()

    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float16, copy=False)
        return value.tolist()
    if isinstance(value, np.floating):
        return np.float16(value).item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    return value


def extract_instances(
    batch_results: List[Any], merge_data_samples_fn: Callable[..., Any]
) -> List[Dict[str, Any]]:
    if not batch_results:
        return []

    data_sample = merge_data_samples_fn(batch_results)
    if "pred_instances" not in data_sample:
        return []

    pred_instances = data_sample.pred_instances
    if "keypoints" not in pred_instances:
        return []

    instances: List[Dict[str, Any]] = []
    num_instances = len(pred_instances.keypoints)

    for idx in range(num_instances):
        instance: Dict[str, Any] = {
            "keypoints": to_serializable(pred_instances.keypoints[idx]),
        }
        if "keypoint_scores" in pred_instances:
            instance["keypoint_scores"] = to_serializable(
                pred_instances.keypoint_scores[idx]
            )
        if "bboxes" in pred_instances:
            instance["bbox"] = to_serializable(pred_instances.bboxes[idx])
        if "bbox_scores" in pred_instances:
            instance["bbox_score"] = to_serializable(pred_instances.bbox_scores[idx])
        instances.append(instance)

    return instances


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = list_images(
        input_dir=input_dir,
        recursive=not args.non_recursive,
        extensions=args.ext,
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {input_dir} with extensions: {args.ext}"
        )

    config_path = str(Path(args.config).expanduser().resolve())
    checkpoint_path = (
        args.checkpoint
        if is_url(args.checkpoint)
        else str(Path(args.checkpoint).expanduser().resolve())
    )

    inference_topdown, init_model, merge_data_samples = import_mmpose_apis()

    # MMPose may resolve dataset metainfo paths like
    # "configs/_base_/datasets/coco.py" relative to CWD.
    with pushd(MMPPOSE_ROOT):
        model = init_model(
            config_path,
            checkpoint_path,
            device=args.device,
        )

    predictions: List[Dict[str, Any]] = []
    for image_path in tqdm(image_paths, desc="Predicting 2D skeletons"):
        try:
            batch_results = inference_topdown(model, str(image_path))
            instances = extract_instances(batch_results, merge_data_samples)
            predictions.append(
                {
                    "image_path": relative_path(image_path, input_dir),
                    "instances": instances,
                }
            )
        except Exception as exc:
            if args.fail_on_error:
                raise
            predictions.append(
                {
                    "image_path": relative_path(image_path, input_dir),
                    "instances": [],
                    "error": str(exc),
                }
            )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dir": str(input_dir),
        "config": config_path,
        "checkpoint": checkpoint_path,
        "device": args.device,
        "num_images": len(image_paths),
        "predictions": predictions,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
