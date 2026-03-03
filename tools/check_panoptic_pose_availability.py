#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEQUENCE_RE = re.compile(r"^[0-9]{6}_[A-Za-z0-9_]+$")


@dataclass
class SequenceAvailability:
    sequence: str
    coco19_http: int
    legacy_http: int
    coco19_tar_local: int
    legacy_tar_local: int
    coco19_json_count: int
    legacy_json_count: int


def _http_status(url: str, timeout_sec: float) -> int:
    req = Request(url, method="HEAD")
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            return int(resp.status)
    except HTTPError as exc:
        return int(exc.code)
    except (URLError, TimeoutError):
        return -1


def _load_sequences(args: argparse.Namespace) -> list[str]:
    sequences: list[str] = []

    if args.list_file is not None:
        if not args.list_file.is_file():
            raise FileNotFoundError(f"--list-file not found: {args.list_file}")
        for raw_line in args.list_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            sequences.append(line)

    if args.toolbox_root is not None:
        if not args.toolbox_root.is_dir():
            raise FileNotFoundError(f"--toolbox-root is not a directory: {args.toolbox_root}")
        for p in sorted(args.toolbox_root.iterdir()):
            if p.is_dir():
                sequences.append(p.name)

    if not sequences:
        raise ValueError("No sequences found. Provide --list-file and/or --toolbox-root.")

    uniq = sorted(set(sequences))
    invalid = [s for s in uniq if SEQUENCE_RE.match(s) is None]
    if invalid:
        raise ValueError(
            "Invalid sequence names found: " + ", ".join(invalid[:10]) + ("..." if len(invalid) > 10 else "")
        )
    return uniq


def _check_one(sequence: str, endpoint_url: str, data_root: Path, timeout_sec: float) -> SequenceAvailability:
    base = endpoint_url.rstrip("/")
    coco19_url = f"{base}/webdata/dataset/{sequence}/hdPose3d_stage1_coco19.tar"
    legacy_url = f"{base}/webdata/dataset/{sequence}/hdPose3d_stage1.tar"

    seq_root = data_root / sequence
    coco19_tar = seq_root / "hdPose3d_stage1_coco19.tar"
    legacy_tar = seq_root / "hdPose3d_stage1.tar"
    coco19_json_root = seq_root / "hdPose3d_stage1_coco19"
    legacy_json_root = seq_root / "hdPose3d_stage1"

    coco19_json_count = 0
    legacy_json_count = 0
    if coco19_json_root.is_dir():
        coco19_json_count = sum(1 for _ in coco19_json_root.rglob("body3DScene_*.json"))
    if legacy_json_root.is_dir():
        legacy_json_count = sum(1 for _ in legacy_json_root.rglob("body3DScene_*.json"))

    return SequenceAvailability(
        sequence=sequence,
        coco19_http=_http_status(coco19_url, timeout_sec=timeout_sec),
        legacy_http=_http_status(legacy_url, timeout_sec=timeout_sec),
        coco19_tar_local=int(coco19_tar.is_file() and coco19_tar.stat().st_size > 0),
        legacy_tar_local=int(legacy_tar.is_file() and legacy_tar.stat().st_size > 0),
        coco19_json_count=coco19_json_count,
        legacy_json_count=legacy_json_count,
    )


def _print_summary(rows: list[SequenceAvailability]) -> None:
    n = len(rows)
    coco19_remote = sum(1 for r in rows if r.coco19_http == 200)
    legacy_remote = sum(1 for r in rows if r.legacy_http == 200)
    either_remote = sum(1 for r in rows if (r.coco19_http == 200 or r.legacy_http == 200))
    coco19_local_tar = sum(r.coco19_tar_local for r in rows)
    legacy_local_tar = sum(r.legacy_tar_local for r in rows)
    coco19_with_json = sum(1 for r in rows if r.coco19_json_count > 0)
    legacy_with_json = sum(1 for r in rows if r.legacy_json_count > 0)
    either_with_json = sum(1 for r in rows if (r.coco19_json_count > 0 or r.legacy_json_count > 0))

    print(f"Total sequences: {n}")
    print(f"Remote available (coco19 tar): {coco19_remote}")
    print(f"Remote available (legacy tar): {legacy_remote}")
    print(f"Remote available (either tar): {either_remote}")
    print(f"Local tar present (coco19): {coco19_local_tar}")
    print(f"Local tar present (legacy): {legacy_local_tar}")
    print(f"Local extracted JSON present (coco19): {coco19_with_json}")
    print(f"Local extracted JSON present (legacy): {legacy_with_json}")
    print(f"Local extracted JSON present (either): {either_with_json}")

    print()
    print("Sequences with neither remote tar available (both non-200):")
    missing = [r.sequence for r in rows if r.coco19_http != 200 and r.legacy_http != 200]
    if not missing:
        print("  (none)")
    else:
        for seq in missing:
            print(f"  - {seq}")


def _write_csv(rows: list[SequenceAvailability], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "coco19_http",
                "legacy_http",
                "coco19_tar_local",
                "legacy_tar_local",
                "coco19_json_count",
                "legacy_json_count",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.sequence,
                    r.coco19_http,
                    r.legacy_http,
                    r.coco19_tar_local,
                    r.legacy_tar_local,
                    r.coco19_json_count,
                    r.legacy_json_count,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Panoptic body pose file availability per sequence "
            "(remote tar + local tar + local extracted body3DScene JSON)."
        )
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help="Text file with one sequence per line.",
    )
    parser.add_argument(
        "--toolbox-root",
        type=Path,
        default=None,
        help="Auto-discover sequence names from immediate subdirectories.",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="http://domedb.perception.cs.cmu.edu",
        help="Dataset endpoint URL.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/shared/multi_view_hpe/panoptic"),
        help="Local root that contains per-sequence directories.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent HTTP checks.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be > 0")

    sequences = _load_sequences(args)
    rows: list[SequenceAvailability] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_check_one, seq, args.endpoint_url, args.data_root, args.timeout_sec): seq
            for seq in sequences
        }
        for fut in as_completed(futs):
            rows.append(fut.result())

    rows.sort(key=lambda x: x.sequence)
    _print_summary(rows)
    if args.output_csv is not None:
        _write_csv(rows, args.output_csv)
        print()
        print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
