#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SequenceSummary:
    name: str
    status: str
    annotation_dir: Path | None
    n_json: int
    n_json_parsed: int
    n_json_parse_error: int
    n_frames_with_body: int
    max_bodies_per_frame: int
    error: str | None = None


def _find_annotation_files(sequence_dir: Path) -> tuple[Path | None, list[Path]]:
    ann_dir = sequence_dir / "hdPose3d_stage1_coco19"
    if not ann_dir.is_dir():
        return None, []
    files = sorted(ann_dir.rglob("body3DScene_*.json"))
    return ann_dir, files


def _summarize_sequence(sequence_dir: Path) -> SequenceSummary:
    ann_dir, files = _find_annotation_files(sequence_dir)
    if ann_dir is None:
        return SequenceSummary(
            name=sequence_dir.name,
            status="missing_annotation_dir",
            annotation_dir=None,
            n_json=0,
            n_json_parsed=0,
            n_json_parse_error=0,
            n_frames_with_body=0,
            max_bodies_per_frame=0,
        )
    if not files:
        return SequenceSummary(
            name=sequence_dir.name,
            status="missing_body3d_json",
            annotation_dir=ann_dir,
            n_json=0,
            n_json_parsed=0,
            n_json_parse_error=0,
            n_frames_with_body=0,
            max_bodies_per_frame=0,
        )

    max_bodies_per_frame = 0
    n_frames_with_body = 0
    n_json_parsed = 0
    n_json_parse_error = 0
    first_error = None
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            bodies = data.get("bodies")
            if not isinstance(bodies, list):
                raise ValueError(
                    f"`bodies` must be a list in {file_path}, got {type(bodies).__name__}"
                )
            body_count = len(bodies)
            if body_count > 0:
                n_frames_with_body += 1
            if body_count > max_bodies_per_frame:
                max_bodies_per_frame = body_count
            n_json_parsed += 1
        except Exception as exc:  # pragma: no cover - defensive path
            n_json_parse_error += 1
            if first_error is None:
                first_error = str(exc)

    if n_json_parsed == 0:
        return SequenceSummary(
            name=sequence_dir.name,
            status="parse_error",
            annotation_dir=ann_dir,
            n_json=len(files),
            n_json_parsed=0,
            n_json_parse_error=n_json_parse_error,
            n_frames_with_body=n_frames_with_body,
            max_bodies_per_frame=max_bodies_per_frame,
            error=first_error,
        )

    if max_bodies_per_frame > 1:
        status = "multi_actor"
    elif max_bodies_per_frame == 1:
        status = "single_actor"
    else:
        status = "no_detected_body"
    if n_json_parse_error > 0:
        status = f"{status}_with_parse_errors"

    return SequenceSummary(
        name=sequence_dir.name,
        status=status,
        annotation_dir=ann_dir,
        n_json=len(files),
        n_json_parsed=n_json_parsed,
        n_json_parse_error=n_json_parse_error,
        n_frames_with_body=n_frames_with_body,
        max_bodies_per_frame=max_bodies_per_frame,
    )


def _collect_sequences(panoptic_root: Path) -> list[Path]:
    if not panoptic_root.is_dir():
        raise FileNotFoundError(f"panoptic root is not a directory: {panoptic_root}")
    return sorted(path for path in panoptic_root.iterdir() if path.is_dir())


def _print_summary(summaries: list[SequenceSummary]) -> None:
    single = [s for s in summaries if s.status.startswith("single_actor")]
    multi = [s for s in summaries if s.status.startswith("multi_actor")]
    unknown = [
        s
        for s in summaries
        if not (s.status.startswith("single_actor") or s.status.startswith("multi_actor"))
    ]

    print(f"Total sequence dirs: {len(summaries)}")
    print(f"Single actor: {len(single)}")
    print(f"Multi actor: {len(multi)}")
    print(f"Unknown/incomplete: {len(unknown)}")
    print()

    print("Single-actor sequences:")
    for item in single:
        print(
            f"  - {item.name} (json={item.n_json}, "
            f"parsed={item.n_json_parsed}, parse_errors={item.n_json_parse_error}, "
            f"frames_with_body={item.n_frames_with_body}, "
            f"max_bodies_per_frame={item.max_bodies_per_frame}, status={item.status})"
        )
    print()

    print("Multi-actor sequences:")
    for item in multi:
        print(
            f"  - {item.name} (json={item.n_json}, "
            f"parsed={item.n_json_parsed}, parse_errors={item.n_json_parse_error}, "
            f"frames_with_body={item.n_frames_with_body}, "
            f"max_bodies_per_frame={item.max_bodies_per_frame}, status={item.status})"
        )
    print()

    if unknown:
        print("Unknown/incomplete sequences:")
        for item in unknown:
            detail = (
                f"status={item.status}, json={item.n_json}, "
                f"parsed={item.n_json_parsed}, parse_errors={item.n_json_parse_error}, "
                f"frames_with_body={item.n_frames_with_body}, "
                f"max_bodies_per_frame={item.max_bodies_per_frame}"
            )
            if item.error:
                detail += f", error={item.error}"
            print(f"  - {item.name} ({detail})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify Panoptic/Kinoptic sequences as single-actor or multi-actor "
            "using body3DScene JSON annotations."
        )
    )
    parser.add_argument(
        "--panoptic-root",
        type=Path,
        default=Path("/data/shared/multi_view_hpe/panoptic"),
        help="Path containing Panoptic sequence directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dirs = _collect_sequences(args.panoptic_root)
    summaries = [_summarize_sequence(sequence_dir) for sequence_dir in sequence_dirs]
    _print_summary(summaries)


if __name__ == "__main__":
    main()
