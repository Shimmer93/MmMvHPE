from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .run_utils import dump_json
except ImportError:  # pragma: no cover
    from run_utils import dump_json


SEGMENT_COLUMNS = [
    "sequence_name",
    "camera_name",
    "segment_index",
    "segment_length",
    "start_frame_id",
    "end_frame_id",
    "num_frames",
    "num_valid_frames",
    "num_invalid_frames",
    "mpjpe_mean",
    "pa_mpjpe_mean",
    "pc_mpjpe_mean",
    "mpjpe_max",
    "pa_mpjpe_max",
    "pc_mpjpe_max",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_segment_logs(run_dir: Path, segment_rows: list[dict]) -> None:
    _write_csv(run_dir / "segments.csv", segment_rows, SEGMENT_COLUMNS)
    dump_json(run_dir / "segments.json", segment_rows)


def group_by_sequence_camera(segment_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in segment_rows:
        grouped[(str(row["sequence_name"]), str(row["camera_name"]))].append(row)

    summary_rows: list[dict] = []
    for (sequence_name, camera_name), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "sequence_name": sequence_name,
                "camera_name": camera_name,
                "num_segments": len(rows),
                "num_valid_frames": int(sum(int(r["num_valid_frames"]) for r in rows)),
                "num_invalid_frames": int(sum(int(r["num_invalid_frames"]) for r in rows)),
                "mpjpe_mean": float(np.nanmean([r["mpjpe_mean"] for r in rows])),
                "pa_mpjpe_mean": float(np.nanmean([r["pa_mpjpe_mean"] for r in rows])),
                "pc_mpjpe_mean": float(np.nanmean([r["pc_mpjpe_mean"] for r in rows])),
                "mpjpe_max": float(np.nanmax([r["mpjpe_max"] for r in rows])),
                "pa_mpjpe_max": float(np.nanmax([r["pa_mpjpe_max"] for r in rows])),
                "pc_mpjpe_max": float(np.nanmax([r["pc_mpjpe_max"] for r in rows])),
            }
        )
    return summary_rows


def write_grouped_summary(run_dir: Path, grouped_rows: list[dict]) -> None:
    fieldnames = [
        "sequence_name",
        "camera_name",
        "num_segments",
        "num_valid_frames",
        "num_invalid_frames",
        "mpjpe_mean",
        "pa_mpjpe_mean",
        "pc_mpjpe_mean",
        "mpjpe_max",
        "pa_mpjpe_max",
        "pc_mpjpe_max",
    ]
    _write_csv(run_dir / "sequence_camera_summary.csv", grouped_rows, fieldnames)
    dump_json(run_dir / "sequence_camera_summary.json", grouped_rows)


def compute_overall_metrics(segment_rows: list[dict]) -> dict:
    frame_rows = []
    for segment in segment_rows:
        frame_rows.extend(segment.get("frame_metrics", []))

    valid_rows = [row for row in frame_rows if bool(row.get("valid", False))]
    invalid_rows = [row for row in frame_rows if not bool(row.get("valid", False))]

    def _mean(metric_name: str) -> float:
        values = [float(row[metric_name]) for row in valid_rows]
        return float(np.mean(values)) if values else float("nan")

    def _max(metric_name: str) -> float:
        values = [float(row[metric_name]) for row in valid_rows]
        return float(np.max(values)) if values else float("nan")

    return {
        "num_segments": int(len(segment_rows)),
        "num_total_frames": int(len(frame_rows)),
        "num_valid_frames": int(len(valid_rows)),
        "num_invalid_frames": int(len(invalid_rows)),
        "mpjpe_mean": _mean("mpjpe"),
        "pa_mpjpe_mean": _mean("pa_mpjpe"),
        "pc_mpjpe_mean": _mean("pc_mpjpe"),
        "mpjpe_max": _max("mpjpe"),
        "pa_mpjpe_max": _max("pa_mpjpe"),
        "pc_mpjpe_max": _max("pc_mpjpe"),
    }


def write_overall_summary(run_dir: Path, overall_metrics: dict) -> None:
    fieldnames = [
        "num_segments",
        "num_total_frames",
        "num_valid_frames",
        "num_invalid_frames",
        "mpjpe_mean",
        "pa_mpjpe_mean",
        "pc_mpjpe_mean",
        "mpjpe_max",
        "pa_mpjpe_max",
        "pc_mpjpe_max",
    ]
    _write_csv(run_dir / "overall_metrics.csv", [overall_metrics], fieldnames)
    dump_json(run_dir / "overall_metrics.json", overall_metrics)


def rank_worst_segments(segment_rows: list[dict], rank_metric: str, top_k: int | None = None) -> list[dict]:
    metric_key = f"{rank_metric}_mean"
    valid_rows = [row for row in segment_rows if np.isfinite(float(row[metric_key]))]
    ranked = sorted(valid_rows, key=lambda row: float(row[metric_key]), reverse=True)
    if top_k is not None:
        return ranked[:top_k]
    return ranked


def write_ranked_summary(run_dir: Path, ranked_rows: list[dict], rank_metric: str) -> None:
    filename_root = f"worst_segments_by_{rank_metric}"
    _write_csv(run_dir / f"{filename_root}.csv", ranked_rows, SEGMENT_COLUMNS)
    dump_json(run_dir / f"{filename_root}.json", ranked_rows)


def write_static_plots(
    run_dir: Path,
    segment_rows: list[dict],
    grouped_rows: list[dict],
    *,
    rank_metric: str,
) -> None:
    metrics = ["mpjpe_mean", "pa_mpjpe_mean", "pc_mpjpe_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric_name in zip(axes, metrics, strict=True):
        values = np.asarray([row[metric_name] for row in segment_rows], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.set_title(f"{metric_name} (no valid segments)")
            ax.axis("off")
            continue
        ax.hist(values, bins=min(30, max(5, values.size // 2)), color="#4C72B0", alpha=0.85)
        ax.set_title(metric_name)
        ax.set_xlabel("meters")
        ax.set_ylabel("segments")
    fig.tight_layout()
    fig.savefig(run_dir / "segment_metric_histograms.png", dpi=160)
    plt.close(fig)

    ordered_rows = sorted(
        segment_rows,
        key=lambda row: (
            str(row["sequence_name"]),
            str(row["camera_name"]),
            int(row["segment_index"]),
        ),
    )
    if ordered_rows:
        x = np.arange(len(ordered_rows), dtype=np.int32)
        mpjpe = np.asarray([row["mpjpe_mean"] for row in ordered_rows], dtype=np.float32)
        pa_mpjpe = np.asarray([row["pa_mpjpe_mean"] for row in ordered_rows], dtype=np.float32)
        pc_mpjpe = np.asarray([row["pc_mpjpe_mean"] for row in ordered_rows], dtype=np.float32)

        fig, ax = plt.subplots(figsize=(max(10, len(ordered_rows) * 0.35), 5.5))
        ax.plot(x, mpjpe, marker="o", linewidth=2, color="#DD8452", label="MPJPE")
        ax.plot(x, pa_mpjpe, marker="o", linewidth=2, color="#4C72B0", label="PA-MPJPE")
        ax.plot(x, pc_mpjpe, marker="o", linewidth=2, color="#55A868", label="PC-MPJPE")
        ax.set_xlabel("Segment Index")
        ax.set_ylabel("Meters")
        ax.set_title("Per-Segment Metrics Over Entire Evaluation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "segment_metric_linechart.png", dpi=160)
        plt.close(fig)

    metric_key = f"{rank_metric}_mean"
    top_rows = sorted(grouped_rows, key=lambda row: float(row[metric_key]), reverse=True)[:15]
    if top_rows:
        labels = [f"{row['sequence_name']}\n{row['camera_name']}" for row in top_rows]
        values = [float(row[metric_key]) for row in top_rows]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
        ax.bar(range(len(labels)), values, color="#DD8452")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("meters")
        ax.set_title(f"Worst sequence-camera pairs by {metric_key}")
        fig.tight_layout()
        fig.savefig(run_dir / f"worst_sequence_camera_pairs_{rank_metric}.png", dpi=160)
        plt.close(fig)
