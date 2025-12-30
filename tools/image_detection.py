import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    YOLO = None
    _YOLO_IMPORT_ERROR = exc


def predict_human_bboxes(
    image_path: str,
    model: Optional[Any] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Predict human bounding boxes in a single image.

    Returns a list of dicts:
    [{"bbox": [x1, y1, x2, y2], "score": float, "class_id": int}, ...]
    """
    if YOLO is None:
        raise ImportError(
            "ultralytics is required for human detection. "
            f"Original error: {_YOLO_IMPORT_ERROR}"
        )

    if model is None:
        default_weights = Path("yolov8n.pt")
        model = YOLO(str(default_weights))

    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    output: List[Dict[str, Any]] = []
    if not results:
        return output

    res = results[0]
    if res.boxes is None or res.boxes.data is None:
        return output

    boxes = res.boxes
    xyxy = boxes.xyxy.tolist()
    scores = boxes.conf.tolist()
    class_ids = boxes.cls.tolist()

    for bbox, score, class_id in zip(xyxy, scores, class_ids):
        class_id_int = int(class_id)
        if class_id_int != 0:
            continue
        output.append(
            {
                "bbox": [float(v) for v in bbox],
                "score": float(score),
                "class_id": class_id_int,
            }
        )

    return output


def annotate_images(
    image_paths: Iterable[str],
    output_path: str,
    model: Optional[Any] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run human detection for a list of image paths and save annotations to JSON.
    Returns the annotation mapping.
    """
    annotations: Dict[str, List[Dict[str, Any]]] = {}

    for image_path in tqdm(image_paths):
        bboxes = predict_human_bboxes(
            image_path=image_path,
            model=model,
            conf=conf,
            iou=iou,
            device=device,
        )
        annotations[str(image_path)] = bboxes

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    return annotations


if __name__ == "__main__":
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(description="Run YOLO human detection on images.")
    parser.add_argument(
        "--input",
        default="data/mmfi/rgb",
        help="Glob pattern or directory for images.",
    )
    parser.add_argument(
        "--output",
        default="data/mmfi/rgb_boxes.json",
        help="Output JSON file path.",
    )
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO weights path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    if args.input.endswith(".jpg") or args.input.endswith(".png"):
        image_paths = [args.input]
    elif Path(args.input).is_dir():
        image_paths = glob(str(Path(args.input) / "*.jpg")) + glob(str(Path(args.input) / "*.png"))
    else:
        image_paths = glob(args.input)

    if not image_paths:
        raise FileNotFoundError(f"No images found for input: {args.input}")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.weights)
    annotate_images(
        image_paths=image_paths,
        output_path=args.output,
        model=model,
        conf=args.conf,
        iou=args.iou,
        device=device,
    )
