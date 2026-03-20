from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class COCOInstanceRecord:
    annotation_id: int
    image_id: int
    image_path: Path
    image_width: int
    image_height: int
    bbox_xywh: np.ndarray
    segmentation: Any
    area: float
    num_keypoints: int
    iscrowd: int


class COCOValPersonAdapter:
    def __init__(
        self,
        data_root: str | Path,
        *,
        annotation_file: str = "annotations/person_keypoints_val2017.json",
        image_dir: str = "val2017",
        min_area: float = 4096.0,
        min_keypoints: int = 5,
        one_person_only: bool = True,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.annotation_path = self.data_root / annotation_file
        self.image_root = self.data_root / image_dir
        self.min_area = float(min_area)
        self.min_keypoints = int(min_keypoints)
        self.one_person_only = bool(one_person_only)
        self._validate_paths()

        payload = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        if not images or not annotations:
            raise ValueError(
                f"COCO annotation file is missing images or annotations: {self.annotation_path}"
            )

        self._image_by_id: dict[int, dict[str, Any]] = {
            int(item["id"]): item for item in images if "id" in item
        }

        eligible: list[dict[str, Any]] = []
        per_image_counts: dict[int, int] = {}
        for ann in annotations:
            if int(ann.get("category_id", -1)) != 1:
                continue
            if int(ann.get("iscrowd", 0)) != 0:
                continue
            if float(ann.get("area", 0.0)) < self.min_area:
                continue
            if int(ann.get("num_keypoints", 0)) < self.min_keypoints:
                continue
            image_id = int(ann["image_id"])
            if image_id not in self._image_by_id:
                continue
            eligible.append(ann)
            per_image_counts[image_id] = per_image_counts.get(image_id, 0) + 1

        if self.one_person_only:
            eligible = [ann for ann in eligible if per_image_counts[int(ann["image_id"])] == 1]

        eligible.sort(key=lambda ann: (int(ann["image_id"]), int(ann["id"])))
        self._records: list[COCOInstanceRecord] = [self._build_record(ann) for ann in eligible]
        if not self._records:
            raise ValueError(
                f"No eligible COCO val person instances found under {self.data_root} "
                f"(one_person_only={self.one_person_only}, min_area={self.min_area}, "
                f"min_keypoints={self.min_keypoints})."
            )

    def _validate_paths(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"COCO data_root not found: {self.data_root}")
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {self.annotation_path}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {self.image_root}")

    def _build_record(self, ann: dict[str, Any]) -> COCOInstanceRecord:
        image_id = int(ann["image_id"])
        image_info = self._image_by_id[image_id]
        image_path = self.image_root / str(image_info["file_name"])
        bbox = np.asarray(ann["bbox"], dtype=np.float32)
        if bbox.shape != (4,):
            raise ValueError(f"Expected bbox shape (4,), got {bbox.shape} for ann={ann['id']}")
        return COCOInstanceRecord(
            annotation_id=int(ann["id"]),
            image_id=image_id,
            image_path=image_path,
            image_width=int(image_info["width"]),
            image_height=int(image_info["height"]),
            bbox_xywh=bbox,
            segmentation=ann.get("segmentation"),
            area=float(ann.get("area", 0.0)),
            num_keypoints=int(ann.get("num_keypoints", 0)),
            iscrowd=int(ann.get("iscrowd", 0)),
        )

    def __len__(self) -> int:
        return len(self._records)

    def get_record(self, index: int) -> COCOInstanceRecord:
        if index < 0 or index >= len(self):
            raise IndexError(f"index={index} out of range for {len(self)} eligible COCO samples.")
        return self._records[index]

    @staticmethod
    def _bbox_xywh_to_xyxy(bbox_xywh: np.ndarray, width: int, height: int) -> np.ndarray:
        x, y, w, h = bbox_xywh.astype(np.float32)
        x1 = np.clip(x, 0.0, max(0, width - 1))
        y1 = np.clip(y, 0.0, max(0, height - 1))
        x2 = np.clip(x + w, x1 + 1.0, width)
        y2 = np.clip(y + h, y1 + 1.0, height)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
        if isinstance(segmentation, list):
            mask = np.zeros((height, width), dtype=np.uint8)
            for poly in segmentation:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 3:
                    continue
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(mask, [pts], color=1)
            return mask

        if isinstance(segmentation, dict):
            try:
                from xtcocotools import mask as mask_utils
            except Exception as exc:
                raise ValueError(
                    "Encountered RLE segmentation but xtcocotools.mask is unavailable."
                ) from exc
            rle = segmentation
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects([rle], height, width)[0]
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[..., 0]
            return (mask > 0).astype(np.uint8)

        raise ValueError(f"Unsupported COCO segmentation format: {type(segmentation).__name__}")

    def load_sample(self, index: int) -> dict[str, Any]:
        record = self.get_record(index)
        if not record.image_path.is_file():
            raise FileNotFoundError(f"COCO image not found: {record.image_path}")

        image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read COCO image: {record.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = self._segmentation_to_mask(
            segmentation=record.segmentation,
            height=record.image_height,
            width=record.image_width,
        )
        if mask.shape != (record.image_height, record.image_width):
            raise ValueError(
                f"Mask shape mismatch for ann={record.annotation_id}: "
                f"{mask.shape} vs {(record.image_height, record.image_width)}"
            )
        bbox_xyxy = self._bbox_xywh_to_xyxy(
            bbox_xywh=record.bbox_xywh,
            width=record.image_width,
            height=record.image_height,
        )
        return {
            "record": record,
            "image_rgb": image_rgb,
            "bbox_xyxy": bbox_xyxy,
            "mask": mask,
            "mask_provenance": "coco_annotation_segmentation",
        }
