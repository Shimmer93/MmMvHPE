#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

SEQUENCE_RE = re.compile(r"^[0-9]{6}_[A-Za-z0-9_]+$")


def parse_sequence_names(
    preprocessed_root: Path,
    sequences_csv: str | None,
    sequence_list: Path | None,
    max_sequences: int | None,
) -> list[str]:
    names: set[str] = set()
    if sequences_csv:
        for item in sequences_csv.split(","):
            seq = item.strip()
            if seq:
                names.add(seq)
    if sequence_list:
        if not sequence_list.is_file():
            raise FileNotFoundError(f"sequence list not found: {sequence_list}")
        for raw in sequence_list.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                names.add(line)
    if not names:
        names = {p.name for p in preprocessed_root.iterdir() if p.is_dir() and SEQUENCE_RE.match(p.name)}

    seqs = sorted(names)
    invalid = [s for s in seqs if SEQUENCE_RE.match(s) is None]
    if invalid:
        raise ValueError(f"invalid sequence names: {invalid[:10]}")
    if max_sequences is not None:
        seqs = seqs[:max_sequences]
    return seqs


def patch_sequence(
    seq_name: str,
    preprocessed_root: Path,
    toolbox_root: Path,
    overwrite: bool,
    backup: bool,
) -> tuple[bool, str]:
    seq_pre = preprocessed_root / seq_name
    cameras_path = seq_pre / "meta" / "cameras_kinect_cropped.json"
    if not cameras_path.is_file():
        return False, f"missing file: {cameras_path}"

    panoptic_calib_path = toolbox_root / seq_name / f"calibration_{seq_name}.json"
    if not panoptic_calib_path.is_file():
        return False, f"missing panoptic calibration: {panoptic_calib_path}"

    with cameras_path.open("r", encoding="utf-8") as f:
        cameras = json.load(f)
    if not isinstance(cameras, dict) or not cameras:
        return False, f"invalid cameras json: {cameras_path}"

    with panoptic_calib_path.open("r", encoding="utf-8") as f:
        panoptic_calib = json.load(f)
    panoptic_cameras = panoptic_calib.get("cameras")
    if not isinstance(panoptic_cameras, list):
        return False, f"invalid cameras in panoptic calibration: {panoptic_calib_path}"
    panoptic_map = {}
    for cam in panoptic_cameras:
        name = cam.get("name")
        if isinstance(name, str):
            panoptic_map[name] = cam

    changed = False
    for cam_key, cam_data in cameras.items():
        if not isinstance(cam_data, dict):
            continue
        if (
            not overwrite
            and "extrinsic_world_to_color" in cam_data
            and "extrinsic_world_to_color_unit" in cam_data
        ):
            continue

        if not cam_key.startswith("kinect_"):
            continue
        try:
            cam_idx = int(cam_key.split("_", 1)[1])
        except ValueError:
            continue

        pan_name = f"50_{cam_idx:02d}"
        if pan_name not in panoptic_map:
            return False, f"camera {pan_name} not found in {panoptic_calib_path}"

        pan_cam = panoptic_map[pan_name]
        r = np.array(pan_cam["R"], dtype=np.float64)
        t = np.array(pan_cam["t"], dtype=np.float64).reshape(3, 1)
        if r.shape != (3, 3):
            return False, f"invalid R shape for {pan_name}: {r.shape}"

        ext = np.hstack((r, t))
        cam_data["extrinsic_world_to_color"] = ext.tolist()
        cam_data["extrinsic_world_to_color_unit"] = "cm"
        cam_data["extrinsic_world_to_color_source"] = f"{panoptic_calib_path.name}:{pan_name}"
        changed = True

    if not changed:
        return True, "already_patched"

    if backup:
        backup_path = cameras_path.with_suffix(".json.bak")
        shutil.copy2(cameras_path, backup_path)

    with cameras_path.open("w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)

    return True, "patched"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch preprocessed Panoptic sequences to embed self-contained world->color extrinsics."
    )
    p.add_argument("--preprocessed-root", type=Path, default=Path("/opt/data/panoptic_kinoptic_single_actor_cropped"))
    p.add_argument("--toolbox-root", type=Path, default=Path("/data/shared/panoptic-toolbox"))
    p.add_argument("--sequences", type=str, default=None, help="Comma-separated sequence names.")
    p.add_argument("--sequence-list", type=Path, default=None, help="Text file with one sequence name per line.")
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing extrinsic_world_to_color fields.")
    p.add_argument("--no-backup", action="store_true", help="Do not create .json.bak backup files.")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.preprocessed_root.is_dir():
        raise FileNotFoundError(f"preprocessed root not found: {args.preprocessed_root}")
    if not args.toolbox_root.is_dir():
        raise FileNotFoundError(f"toolbox root not found: {args.toolbox_root}")

    seqs = parse_sequence_names(
        preprocessed_root=args.preprocessed_root,
        sequences_csv=args.sequences,
        sequence_list=args.sequence_list,
        max_sequences=args.max_sequences,
    )
    if not seqs:
        raise ValueError("no sequences selected")

    ok_count = 0
    patched_count = 0
    failed: list[tuple[str, str]] = []

    for seq_name in tqdm(seqs, desc="Patch sequences"):
        try:
            ok, msg = patch_sequence(
                seq_name=seq_name,
                preprocessed_root=args.preprocessed_root,
                toolbox_root=args.toolbox_root,
                overwrite=args.overwrite,
                backup=not args.no_backup,
            )
            if ok:
                ok_count += 1
                if msg == "patched":
                    patched_count += 1
                print(f"[patch-panoptic-extrinsics] {seq_name}: {msg}")
            else:
                failed.append((seq_name, msg))
                print(f"[patch-panoptic-extrinsics] {seq_name}: FAILED: {msg}")
                if not args.continue_on_error:
                    raise RuntimeError(msg)
        except Exception as exc:
            failed.append((seq_name, str(exc)))
            print(f"[patch-panoptic-extrinsics] {seq_name}: FAILED: {exc}")
            if not args.continue_on_error:
                raise

    print(
        f"[patch-panoptic-extrinsics] completed. total={len(seqs)} ok={ok_count} patched={patched_count} failed={len(failed)}"
    )
    for seq_name, msg in failed:
        print(f"[patch-panoptic-extrinsics] failed: {seq_name}: {msg}")


if __name__ == "__main__":
    main()
