#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SEQ_RE = re.compile(r"^[0-9]{6}_[A-Za-z0-9_]+$")


@dataclass
class SeqStats:
    seq: str
    total: int = 0
    hip_bad: int = 0
    neck_center_bad: int = 0
    hip_min: float = float("inf")
    hip_med: float = 0.0
    neck_center_min: float = float("inf")
    neck_center_med: float = 0.0
    hip_low_conf: int = 0
    neck_center_low_conf: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Check Panoptic joints19 degeneracy in preprocessed gt3d files. "
            "Measures ||rHip-lHip|| and ||neck-bodyCenter||."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/opt/data/panoptic_kinoptic_single_actor_cropped"),
        help="Preprocessed Panoptic root directory.",
    )
    p.add_argument(
        "--hip-thresh-cm",
        type=float,
        default=1.0,
        help="Threshold (cm) for considering ||rHip-lHip|| degenerate.",
    )
    p.add_argument(
        "--neck-center-thresh-cm",
        type=float,
        default=1.0,
        help="Threshold (cm) for considering ||neck-bodyCenter|| degenerate.",
    )
    p.add_argument(
        "--conf-thresh",
        type=float,
        default=0.1,
        help="Confidence threshold for flagging low-confidence joints.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many sequences to print (sorted by hip degeneracy rate).",
    )
    return p.parse_args()


def list_sequences(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"root not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir() and SEQ_RE.match(p.name))


def main() -> None:
    args = parse_args()
    seq_dirs = list_sequences(args.root)
    if not seq_dirs:
        raise ValueError(f"no sequence directories found under {args.root}")

    hip_thr = float(args.hip_thresh_cm)
    nc_thr = float(args.neck_center_thresh_cm)
    conf_thr = float(args.conf_thresh)

    all_stats: list[SeqStats] = []

    global_total = 0
    global_hip_bad = 0
    global_nc_bad = 0
    global_hip_low_conf = 0
    global_nc_low_conf = 0

    for seq_dir in seq_dirs:
        gt_dir = seq_dir / "gt3d"
        if not gt_dir.is_dir():
            continue
        gt_files = sorted(gt_dir.glob("*.npy"))
        if not gt_files:
            continue

        stats = SeqStats(seq=seq_dir.name)
        hip_vals = []
        nc_vals = []

        for f in gt_files:
            arr = np.load(f)  # expected (19,4): x,y,z,score in cm
            if arr.ndim != 2 or arr.shape[0] < 13 or arr.shape[1] < 3:
                continue

            xyz = arr[:, :3].astype(np.float32)
            conf = arr[:, 3].astype(np.float32) if arr.shape[1] >= 4 else None

            # Panoptic joints19:
            # 0 neck, 2 bodyCenter, 6 lHip, 12 rHip
            neck = xyz[0]
            body = xyz[2]
            lhip = xyz[6]
            rhip = xyz[12]

            hip_d = float(np.linalg.norm(rhip - lhip))
            nc_d = float(np.linalg.norm(neck - body))

            hip_vals.append(hip_d)
            nc_vals.append(nc_d)
            stats.total += 1

            if hip_d < hip_thr:
                stats.hip_bad += 1
            if nc_d < nc_thr:
                stats.neck_center_bad += 1

            if conf is not None:
                if min(float(conf[6]), float(conf[12])) < conf_thr:
                    stats.hip_low_conf += 1
                if min(float(conf[0]), float(conf[2])) < conf_thr:
                    stats.neck_center_low_conf += 1

        if stats.total == 0:
            continue

        stats.hip_min = float(np.min(hip_vals))
        stats.hip_med = float(np.median(hip_vals))
        stats.neck_center_min = float(np.min(nc_vals))
        stats.neck_center_med = float(np.median(nc_vals))

        all_stats.append(stats)
        global_total += stats.total
        global_hip_bad += stats.hip_bad
        global_nc_bad += stats.neck_center_bad
        global_hip_low_conf += stats.hip_low_conf
        global_nc_low_conf += stats.neck_center_low_conf

    if not all_stats:
        raise ValueError("no valid gt3d frames found")

    all_stats.sort(
        key=lambda s: (s.hip_bad / s.total if s.total > 0 else 0.0),
        reverse=True,
    )

    print(f"root: {args.root}")
    print(f"sequences_checked: {len(all_stats)}")
    print(
        "global: "
        f"frames={global_total} "
        f"hip_bad(<{hip_thr:.3f}cm)={global_hip_bad} ({(100.0*global_hip_bad/max(global_total,1)):.4f}%) "
        f"neck_center_bad(<{nc_thr:.3f}cm)={global_nc_bad} ({(100.0*global_nc_bad/max(global_total,1)):.4f}%)"
    )
    print(
        "global low-confidence: "
        f"hip_pairs(min_conf<{conf_thr:.3f})={global_hip_low_conf} ({(100.0*global_hip_low_conf/max(global_total,1)):.4f}%) "
        f"neck_center_pairs(min_conf<{conf_thr:.3f})={global_nc_low_conf} ({(100.0*global_nc_low_conf/max(global_total,1)):.4f}%)"
    )
    print()
    print(
        "per-sequence (sorted by hip degeneracy rate):\n"
        "seq total hip_bad_rate hip_bad hip_min_cm hip_med_cm "
        "neck_center_bad_rate neck_center_bad nc_min_cm nc_med_cm "
        "hip_low_conf_rate nc_low_conf_rate"
    )
    for s in all_stats[: args.top_k]:
        hip_bad_rate = 100.0 * s.hip_bad / max(s.total, 1)
        nc_bad_rate = 100.0 * s.neck_center_bad / max(s.total, 1)
        hip_low_rate = 100.0 * s.hip_low_conf / max(s.total, 1)
        nc_low_rate = 100.0 * s.neck_center_low_conf / max(s.total, 1)
        print(
            f"{s.seq} {s.total} "
            f"{hip_bad_rate:.4f}% {s.hip_bad} {s.hip_min:.4f} {s.hip_med:.4f} "
            f"{nc_bad_rate:.4f}% {s.neck_center_bad} {s.neck_center_min:.4f} {s.neck_center_med:.4f} "
            f"{hip_low_rate:.4f}% {nc_low_rate:.4f}%"
        )


if __name__ == "__main__":
    main()
