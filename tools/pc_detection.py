import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
except ImportError as exc:  # pragma: no cover
    cfg = None
    cfg_from_yaml_file = None
    DatasetTemplate = object
    build_network = None
    load_data_to_gpu = None
    common_utils = None
    _PCDET_IMPORT_ERROR = exc


class NpyPointCloudDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path: Optional[Path] = None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=root_path,
            logger=None,
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise NotImplementedError("Use prepare_sample for single-frame inference.")

    def prepare_sample(self, points: np.ndarray, frame_id: str):
        data_dict = {
            "points": points,
            "frame_id": frame_id,
        }
        return self.prepare_data(data_dict)


def build_openpcdet_model(
    config_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[Any, NpyPointCloudDataset, List[str]]:
    if cfg_from_yaml_file is None or build_network is None:
        raise ImportError(
            "OpenPCDet is required for point cloud detection. "
            f"Original error: {_PCDET_IMPORT_ERROR}"
        )

    cfg_from_yaml_file(config_path, cfg)
    logger = common_utils.create_logger()

    dataset = NpyPointCloudDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(cfg.DATA_CONFIG.get("DATA_PATH", ".")),
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(
        filename=checkpoint_path,
        logger=logger,
        to_cpu=device == "cpu",
    )
    model.eval()
    if device is not None:
        model.to(device)

    return model, dataset, list(cfg.CLASS_NAMES)


def _load_npy_points(npy_path: str) -> np.ndarray:
    points = np.load(npy_path)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("Expected point cloud shape (N, 3) or (N, 4).")
    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1), dtype=points.dtype)
        points = np.concatenate([points, intensity], axis=1)
    return points.astype(np.float32)


def predict_human_bboxes_pc(
    npy_path: str,
    model: Any,
    dataset: NpyPointCloudDataset,
    class_name: str = "Pedestrian",
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    points = _load_npy_points(npy_path)
    frame_id = Path(npy_path).stem

    data_dict = dataset.prepare_sample(points, frame_id)
    data_dict = dataset.collate_batch([data_dict])
    if device is not None:
        data_dict["batch_size"] = len(data_dict["frame_id"])
    load_data_to_gpu(data_dict)

    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)

    pred = pred_dicts[0]
    boxes = pred["pred_boxes"].cpu().numpy()
    scores = pred["pred_scores"].cpu().numpy()
    labels = pred["pred_labels"].cpu().numpy()

    output: List[Dict[str, Any]] = []
    class_names = dataset.class_names
    for box, score, label in zip(boxes, scores, labels):
        label_idx = int(label) - 1
        label_name = class_names[label_idx]
        if label_name != class_name:
            continue
        output.append(
            {
                "bbox": [float(v) for v in box.tolist()],
                "score": float(score),
                "class_id": int(label),
                "class_name": label_name,
            }
        )

    return output


def annotate_pointclouds(
    npy_paths: Iterable[str],
    output_path: str,
    model: Any,
    dataset: NpyPointCloudDataset,
    class_name: str = "Pedestrian",
    device: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    annotations: Dict[str, List[Dict[str, Any]]] = {}

    for npy_path in tqdm(npy_paths):
        bboxes = predict_human_bboxes_pc(
            npy_path=npy_path,
            model=model,
            dataset=dataset,
            class_name=class_name,
            device=device,
        )
        annotations[str(npy_path)] = bboxes

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    return annotations

if __name__ == '__main__':
    import argparse
    from glob import glob
    import os

    parser = argparse.ArgumentParser(description="Run OpenPCDet on .npy point clouds.")
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "PCDET_CONFIG",
            "weights/pc_detection/pointpillar.yaml",
        ),
        help="Path to OpenPCDet config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PCDET_CHECKPOINT", "weights/pc_detection/pointpillar_7728.pth"),
        help="Path to OpenPCDet checkpoint.",
    )
    parser.add_argument(
        "--input",
        default="data/mmfi/lidar",
        help="Glob pattern or directory for .npy files.",
    )
    parser.add_argument(
        "--output",
        default="data/mmfi/lidar_boxes.json",
        help="Output JSON file path.",
    )
    parser.add_argument("--class-name", default="Pedestrian", help="Target class name.")
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            "Set --config or PCDET_CONFIG."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Set --checkpoint or PCDET_CHECKPOINT."
        )

    if args.input.endswith(".npy"):
        npy_paths = [args.input]
    elif Path(args.input).is_dir():
        npy_paths = glob(str(Path(args.input) / "*.npy"))
    else:
        npy_paths = glob(args.input)

    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found for input: {args.input}")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, dataset, _ = build_openpcdet_model(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    annotate_pointclouds(
        npy_paths=npy_paths,
        output_path=args.output,
        model=model,
        dataset=dataset,
        class_name=args.class_name,
        device=device,
    )
