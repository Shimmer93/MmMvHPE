#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import importlib.util
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'third_party' / 'sam-3d-body'))

from misc.official_mhr_smpl_conversion import (
    DEFAULT_MHR_ROOT,
    DEFAULT_SMPL_MODEL_PATH,
    OfficialSam3dToSmplConverter,
)
from misc.sam3d_eval import sam3_cam_int_from_rgb_camera
from misc.registry import create_dataset
from misc.utils import load_cfg, merge_args_cfg
from metrics.mpjpe import mpjpe_func, pampjpe_func, pcmpjpe_func


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PANOPTIC_RUN_SEGMENT_EVAL = _load_module(
    "sam3d_panoptic_run_segment_eval",
    REPO_ROOT / "tools" / "sam3d_panoptic_segment_eval" / "run_segment_eval.py",
)
_PANOPTIC_JOINT_ADAPTER = _load_module(
    "sam3d_panoptic_joint_adapter",
    REPO_ROOT / "tools" / "sam3d_panoptic_segment_eval" / "joint_adapter.py",
)

_load_estimator = _PANOPTIC_RUN_SEGMENT_EVAL._load_estimator
_process_image_for_display = _PANOPTIC_RUN_SEGMENT_EVAL._process_image_for_display
_world_to_camera = _PANOPTIC_RUN_SEGMENT_EVAL._world_to_camera
SAM3ToPanopticCOCO19Adapter = _PANOPTIC_JOINT_ADAPTER.SAM3ToPanopticCOCO19Adapter


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


SCENARIOS = {
    "humman_cross_camera_test": {
        "cfg": "configs/exp/humman/cross_camera_split/hpe.yml",
        "split": "test",
        "dataset_kind": "humman",
    },
    "panoptic_cross_camera_small_test": {
        "cfg": "configs/exp/panoptic/cross_camera_split/xfi_small.yml",
        "split": "test",
        "dataset_kind": "panoptic",
    },
    "panoptic_temporal_occluded_test": {
        "cfg": "configs/baseline/occlusion_robustness/panoptic_seq1_panoptic_final_eval_occluded.yml",
        "split": "test",
        "dataset_kind": "panoptic",
    },
    "panoptic_temporal_unoccluded_test": {
        "cfg": "configs/baseline/occlusion_robustness/panoptic_seq1_panoptic_final_eval_unoccluded.yml",
        "split": "test",
        "dataset_kind": "panoptic",
    },
}


@dataclass(frozen=True)
class ScenarioContext:
    scenario_name: str
    cfg_path: str
    split: str
    camera: str
    dataset_kind: str
    hparams: Any
    dataset_cfg: dict
    pipeline_cfg: list
    dataset: Any
    denorm_params: dict | None


class _MockArgs:
    checkpoint_path = ""
    gpus = 1
    num_workers = 0
    batch_size = 1
    batch_size_eva = 1
    pin_memory = False
    prefetch_factor = 2
    use_wandb = False
    save_test_preds = False


@dataclass(frozen=True)
class FrameMetricResult:
    sample_index: int
    frame_id: int
    valid: bool
    reason: str | None
    mpjpe: float
    pa_mpjpe: float
    pc_mpjpe: float


def _invalid_frame(sample_index: int, frame_id: int, reason: str) -> FrameMetricResult:
    return FrameMetricResult(sample_index, frame_id, False, reason, float('nan'), float('nan'), float('nan'))


def _evaluate_frame_metrics(pred_keypoints: np.ndarray, gt_keypoints: np.ndarray, *, sample_index: int, frame_id: int, pelvis_idx: int, expected_num_joints: int) -> FrameMetricResult:
    pred = np.asarray(pred_keypoints, dtype=np.float32)
    gt = np.asarray(gt_keypoints, dtype=np.float32)
    if pred.shape != (expected_num_joints, 3):
        raise ValueError(f"Predicted joints must have shape ({expected_num_joints},3), got {pred.shape}.")
    if gt.shape != (expected_num_joints, 3):
        raise ValueError(f"GT joints must have shape ({expected_num_joints},3), got {gt.shape}.")
    if not np.isfinite(pred).all():
        return _invalid_frame(sample_index, frame_id, "non_finite_prediction")
    if not np.isfinite(gt).all():
        return _invalid_frame(sample_index, frame_id, "non_finite_ground_truth")
    pred_batch = pred[None, ...]
    gt_batch = gt[None, ...]
    return FrameMetricResult(
        sample_index=sample_index,
        frame_id=frame_id,
        valid=True,
        reason=None,
        mpjpe=float(mpjpe_func(pred_batch, gt_batch, reduce=True)),
        pa_mpjpe=float(pampjpe_func(pred_batch, gt_batch, reduce=True)),
        pc_mpjpe=float(pcmpjpe_func(pred_batch, gt_batch, pelvis_idx=pelvis_idx, reduce=True)),
    )


def _resolve_dataset_cfg(hparams: Any, split: str) -> tuple[dict, list]:
    if split == 'train':
        return hparams.train_dataset, hparams.train_pipeline
    if split == 'val':
        return hparams.val_dataset, hparams.val_pipeline
    if split == 'test':
        return hparams.test_dataset, hparams.test_pipeline
    raise ValueError(f"Unsupported split={split}")


def _denorm_from_pipeline(hparams: Any, pipeline_cfg: list) -> dict | None:
    params = getattr(hparams, 'vis_denorm_params', None)
    if params is not None:
        return params
    for step in pipeline_cfg:
        if not isinstance(step, dict):
            continue
        if step.get('name') != 'VideoNormalize':
            continue
        step_params = step.get('params') or {}
        if step_params.get('norm_mode') == 'imagenet':
            return {
                'rgb_mean': [123.675, 116.28, 103.53],
                'rgb_std': [58.395, 57.12, 57.375],
                'depth_mean': [0.0],
                'depth_std': [255.0],
            }
    return None


def _build_context(scenario_name: str, cfg_path: str, split: str, dataset_kind: str, camera: str) -> ScenarioContext:
    cfg = load_cfg(cfg_path)
    hparams = merge_args_cfg(_MockArgs(), cfg)
    dataset_cfg, pipeline_cfg = _resolve_dataset_cfg(hparams, split)
    dataset_cfg = copy.deepcopy(dataset_cfg)
    pipeline_cfg = copy.deepcopy(pipeline_cfg)
    params = copy.deepcopy(dataset_cfg['params'])

    modality_names = [str(x).lower() for x in params.get('modality_names', [])]
    if 'rgb' not in modality_names:
        raise ValueError(f"Scenario {scenario_name} requires RGB, got modality_names={modality_names}.")
    if int(params.get('seq_len', 1)) != 1:
        raise ValueError(f"Scenario {scenario_name} requires seq_len=1, got {params.get('seq_len')}.")

    params['rgb_cameras_per_sample'] = 1
    params['rgb_cameras'] = [camera]
    if 'depth_cameras' in params:
        params['depth_cameras'] = [camera]
        params['depth_cameras_per_sample'] = 1
        params['lidar_cameras_per_sample'] = 1
    if dataset_kind == 'humman':
        params['use_all_pairs'] = False
    dataset_cfg['params'] = params

    dataset, _ = create_dataset(dataset_cfg['name'], dataset_cfg['params'], pipeline_cfg)
    if len(dataset) == 0:
        raise ValueError(f"Camera {camera} has no samples for scenario {scenario_name}.")

    return ScenarioContext(
        scenario_name=scenario_name,
        cfg_path=str(Path(cfg_path).expanduser().resolve()),
        split=split,
        camera=camera,
        dataset_kind=dataset_kind,
        hparams=hparams,
        dataset_cfg=dataset_cfg,
        pipeline_cfg=pipeline_cfg,
        dataset=dataset,
        denorm_params=_denorm_from_pipeline(hparams, pipeline_cfg),
    )


class PanopticAdapterWrapper:
    def __init__(self) -> None:
        self._adapter = SAM3ToPanopticCOCO19Adapter()
        self.num_joints = 19
        self.pelvis_idx = 2

    def adapt(self, sam_keypoints_3d: np.ndarray) -> np.ndarray:
        return self._adapter.adapt(sam_keypoints_3d)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_rgb_camera(sample: dict) -> dict:
    rgb_camera = sample['rgb_camera']
    if isinstance(rgb_camera, list):
        if len(rgb_camera) != 1:
            raise ValueError(f"Expected one rgb_camera entry, got {len(rgb_camera)}.")
        rgb_camera = rgb_camera[0]
    return rgb_camera


def _extract_frame_id(ctx: ScenarioContext, sample_index: int) -> int:
    data_info = ctx.dataset.data_list[sample_index]
    if ctx.dataset_kind == 'panoptic':
        frame_ids = list(ctx.dataset.sequence_data[data_info['seq_name']]['frame_ids'])
        start_frame = int(data_info['start_frame'])
        return int(frame_ids[start_frame])
    return int(data_info['start_frame'])


def _run_estimator_one_image(*, estimator, rgb_image, rgb_camera: dict, use_mask: bool) -> tuple[list[dict[str, Any]] | None, str | None]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            outputs = estimator.process_one_image(
                rgb_image,
                cam_int=sam3_cam_int_from_rgb_camera(rgb_camera),
                use_mask=use_mask,
            )
        return outputs, None
    except IndexError as exc:
        message = str(exc)
        if "index 0 is out of bounds for axis 0 with size 0" in message:
            return None, "empty_mask_output"
        raise
    except Exception as exc:  # noqa: BLE001
        captured = (stdout_buffer.getvalue() + "\n" + stderr_buffer.getvalue()).strip()
        detail = f"{type(exc).__name__}: {exc}"
        if captured:
            detail = f"{detail} | sam3d_log={captured.splitlines()[-1]}"
        return None, detail


def _group_samples_by_pair(ctx: ScenarioContext) -> list[tuple[str, str, list[tuple[int, int]]]]:
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for sample_idx, data_info in enumerate(ctx.dataset.data_list):
        seq_name = str(data_info['seq_name'])
        rgb_cams = list(data_info.get('rgb_cameras', []))
        if ctx.camera not in rgb_cams:
            continue
        frame_id = _extract_frame_id(ctx, sample_idx)
        grouped[(seq_name, ctx.camera)].append((sample_idx, frame_id))
    rows = []
    for (seq_name, camera_name), items in sorted(grouped.items()):
        rows.append((seq_name, camera_name, sorted(items, key=lambda x: x[1])))
    return rows


def _aggregate_frame_results(frame_results: list[FrameMetricResult]) -> dict:
    valid = [x for x in frame_results if x.valid]
    def _mean(name: str) -> float:
        vals = [float(getattr(x, name)) for x in valid]
        return float(np.mean(vals)) if vals else float('nan')
    def _max(name: str) -> float:
        vals = [float(getattr(x, name)) for x in valid]
        return float(np.max(vals)) if vals else float('nan')
    return {
        'num_total_frames': int(len(frame_results)),
        'num_valid_frames': int(len(valid)),
        'num_invalid_frames': int(len(frame_results) - len(valid)),
        'mpjpe_mean': _mean('mpjpe'),
        'pa_mpjpe_mean': _mean('pa_mpjpe'),
        'pc_mpjpe_mean': _mean('pc_mpjpe'),
        'mpjpe_max': _max('mpjpe'),
        'pa_mpjpe_max': _max('pa_mpjpe'),
        'pc_mpjpe_max': _max('pc_mpjpe'),
        'frame_metrics': [
            {
                'sample_index': int(x.sample_index),
                'frame_id': int(x.frame_id),
                'valid': bool(x.valid),
                'reason': x.reason,
                'mpjpe': float(x.mpjpe),
                'pa_mpjpe': float(x.pa_mpjpe),
                'pc_mpjpe': float(x.pc_mpjpe),
            }
            for x in frame_results
        ],
    }


def _evaluate_one_frame(*, ctx: ScenarioContext, sample_index: int, frame_id: int, estimator, joint_adapter, use_mask: bool, invalid_frame_mode: str):
    sample = ctx.dataset[sample_index]
    selected_rgb = list(sample.get('selected_cameras', {}).get('rgb', []))
    if selected_rgb != [ctx.camera]:
        raise ValueError(f"Sample camera mismatch at sample_index={sample_index}: expected {[ctx.camera]}, got {selected_rgb}.")
    rgb_image = _process_image_for_display(sample['input_rgb'], ctx.denorm_params, key='rgb')
    outputs, error_reason = _run_estimator_one_image(
        estimator=estimator,
        rgb_image=rgb_image,
        rgb_camera=_extract_rgb_camera(sample),
        use_mask=use_mask,
    )
    if outputs is None:
        if invalid_frame_mode == 'error':
            raise RuntimeError(
                f"Invalid SAM3 output for sample_index={sample_index}, frame_id={frame_id}: {error_reason}"
            )
        return _invalid_frame(sample_index, frame_id, str(error_reason))
    if len(outputs) != 1:
        reason = f"expected_single_person_got_{len(outputs)}"
        if invalid_frame_mode == 'error':
            raise RuntimeError(f"Invalid SAM3 output for sample_index={sample_index}, frame_id={frame_id}: {reason}")
        return _invalid_frame(sample_index, frame_id, reason)

    person = outputs[0]
    pred_joints_3d = np.asarray(person['pred_keypoints_3d'], dtype=np.float32)
    pred_cam_t = np.asarray(person['pred_cam_t'], dtype=np.float32).reshape(1, 3)
    pred_camera = joint_adapter.adapt(pred_joints_3d + pred_cam_t)

    gt_world = _to_numpy(sample['gt_keypoints']).astype(np.float32)
    rgb_camera = _extract_rgb_camera(sample)
    gt_camera = _world_to_camera(gt_world, rgb_camera['extrinsic'])
    return _evaluate_frame_metrics(
        pred_camera,
        gt_camera,
        sample_index=sample_index,
        frame_id=frame_id,
        pelvis_idx=joint_adapter.pelvis_idx,
        expected_num_joints=joint_adapter.num_joints,
    )


def _evaluate_humman_pair(
    *,
    ctx: ScenarioContext,
    items: list[tuple[int, int]],
    estimator,
    converter: OfficialSam3dToSmplConverter,
    use_mask: bool,
    invalid_frame_mode: str,
) -> list[FrameMetricResult]:
    frame_results: list[FrameMetricResult] = []
    pending_outputs: list[dict[str, Any]] = []
    pending_meta: list[dict[str, Any]] = []
    for sample_index, frame_id in items:
        sample = ctx.dataset[int(sample_index)]
        selected_rgb = list(sample.get('selected_cameras', {}).get('rgb', []))
        if selected_rgb != [ctx.camera]:
            raise ValueError(
                f"Sample camera mismatch at sample_index={sample_index}: expected {[ctx.camera]}, got {selected_rgb}."
            )
        rgb_image = _process_image_for_display(sample['input_rgb'], ctx.denorm_params, key='rgb')
        outputs, error_reason = _run_estimator_one_image(
            estimator=estimator,
            rgb_image=rgb_image,
            rgb_camera=_extract_rgb_camera(sample),
            use_mask=use_mask,
        )
        if outputs is None:
            if invalid_frame_mode == 'error':
                raise RuntimeError(
                    f"Invalid SAM3 output for sample_index={sample_index}, frame_id={frame_id}: {error_reason}"
                )
            frame_results.append(_invalid_frame(int(sample_index), int(frame_id), str(error_reason)))
            continue
        if len(outputs) != 1:
            reason = f"expected_single_person_got_{len(outputs)}"
            if invalid_frame_mode == 'error':
                raise RuntimeError(
                    f"Invalid SAM3 output for sample_index={sample_index}, frame_id={frame_id}: {reason}"
                )
            frame_results.append(_invalid_frame(int(sample_index), int(frame_id), reason))
            continue
        rgb_camera = _extract_rgb_camera(sample)
        gt_world = _to_numpy(sample['gt_keypoints']).astype(np.float32)
        gt_camera = _world_to_camera(gt_world, rgb_camera['extrinsic'])
        pending_outputs.append(outputs[0])
        pending_meta.append(
            {
                'sample_index': int(sample_index),
                'frame_id': int(frame_id),
                'gt_camera': gt_camera,
            }
        )

    if not pending_outputs:
        return frame_results

    converted = converter.convert_outputs(pending_outputs)
    pred_batch = np.asarray(converted['smpl_joints24'], dtype=np.float32)
    if pred_batch.shape[0] != len(pending_meta):
        raise ValueError(
            f"Official conversion returned {pred_batch.shape[0]} predictions for {len(pending_meta)} inputs."
        )

    for pred_camera, meta in zip(pred_batch, pending_meta):
        frame_results.append(
            _evaluate_frame_metrics(
                pred_camera,
                meta['gt_camera'],
                sample_index=meta['sample_index'],
                frame_id=meta['frame_id'],
                pelvis_idx=0,
                expected_num_joints=24,
            )
        )
    return frame_results


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _scenario_camera_list(cfg_path: str, split: str) -> list[str]:
    cfg = load_cfg(cfg_path)
    hparams = merge_args_cfg(_MockArgs(), cfg)
    dataset_cfg, _ = _resolve_dataset_cfg(hparams, split)
    return [str(x) for x in dataset_cfg['params'].get('rgb_cameras', [])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate SAM3D on the requested HuMMan/Panoptic test suites.', allow_abbrev=False)
    parser.add_argument('--scenarios', default=','.join(SCENARIOS.keys()), help='Comma-separated scenario keys')
    parser.add_argument('--checkpoint-root', default='/opt/data/SAM_3dbody_checkpoints', help='SAM-3D-Body checkpoint root')
    parser.add_argument('--device', default='cuda', help='Inference device')
    parser.add_argument('--segmentor-name', default='none', choices=['none', 'sam2', 'sam3'], help='Optional human segmentor backend')
    parser.add_argument('--segmentor-path', default='/opt/data/SAM3_checkpoint', help='Segmentor checkpoint path')
    parser.add_argument('--mhr-root', default=str(DEFAULT_MHR_ROOT), help='Official MHR checkout root')
    parser.add_argument('--smpl-model-path', default=str(DEFAULT_SMPL_MODEL_PATH), help='SMPL model path for official HuMMan conversion')
    parser.add_argument('--conversion-batch-size', type=int, default=256, help='Official MHR-to-SMPL conversion batch size')
    parser.add_argument('--use-mask', action='store_true', help='Enable mask-conditioned SAM3D inference')
    parser.add_argument('--invalid-frame-mode', default='drop', choices=['drop', 'error'], help='How to handle invalid frames')
    parser.add_argument('--output-root', default='logs/sam3d_eval_suite', help='Root directory for outputs')
    parser.add_argument('--run-name', default=None, help='Optional run directory name')
    parser.add_argument('--max-pairs-per-scenario', type=int, default=None, help='Optional debug cap on sequence-camera pairs per scenario')
    parser.add_argument('--max-frames-per-pair', type=int, default=None, help='Optional debug cap on frames per pair')
    parser.add_argument('--frame-subsample-stride', type=int, default=1, help='Keep every Nth frame in each sequence-camera pair (1 means no subsampling)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_names = [x.strip() for x in str(args.scenarios).split(',') if x.strip()]
    unknown = [name for name in scenario_names if name not in SCENARIOS]
    if unknown:
        raise ValueError(f'Unknown scenarios: {unknown}. Known: {sorted(SCENARIOS)}')
    if int(args.frame_subsample_stride) <= 0:
        raise ValueError(f'frame_subsample_stride must be >= 1, got {args.frame_subsample_stride}')

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"suite_{'mask' if args.use_mask else 'nomask'}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    estimator = _load_estimator(Path(args.checkpoint_root).expanduser().resolve(), args.device, args.segmentor_name, args.segmentor_path)
    panoptic_adapter = PanopticAdapterWrapper()
    humman_converter = None
    if any(SCENARIOS[name]['dataset_kind'] == 'humman' for name in scenario_names):
        humman_converter = OfficialSam3dToSmplConverter(
            device=args.device,
            mhr_root=args.mhr_root,
            smpl_model_path=args.smpl_model_path,
            batch_size=args.conversion_batch_size,
        )

    suite_summary = []
    pair_fieldnames = [
        'scenario', 'dataset_kind', 'sequence_name', 'camera_name', 'num_total_frames', 'num_valid_frames', 'num_invalid_frames',
        'mpjpe_mean', 'pa_mpjpe_mean', 'pc_mpjpe_mean', 'mpjpe_max', 'pa_mpjpe_max', 'pc_mpjpe_max', 'cfg_path'
    ]
    camera_fieldnames = [
        'scenario', 'dataset_kind', 'camera_name', 'num_pairs', 'num_total_frames', 'num_valid_frames', 'num_invalid_frames',
        'mpjpe_mean', 'pa_mpjpe_mean', 'pc_mpjpe_mean', 'mpjpe_max', 'pa_mpjpe_max', 'pc_mpjpe_max'
    ]
    overall_fieldnames = [
        'scenario', 'dataset_kind', 'num_pairs', 'num_total_frames', 'num_valid_frames', 'num_invalid_frames',
        'mpjpe_mean', 'pa_mpjpe_mean', 'pc_mpjpe_mean', 'mpjpe_max', 'pa_mpjpe_max', 'pc_mpjpe_max'
    ]

    for scenario_name in scenario_names:
        spec = SCENARIOS[scenario_name]
        scenario_dir = run_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        camera_names = _scenario_camera_list(spec['cfg'], spec['split'])
        pair_rows: list[dict] = []
        skipped_cameras: list[dict] = []
        scenario_pair_count = 0

        for camera_name in tqdm(camera_names, desc=f'{scenario_name}:cameras'):
            if args.max_pairs_per_scenario is not None and scenario_pair_count >= args.max_pairs_per_scenario:
                break
            try:
                ctx = _build_context(scenario_name, spec['cfg'], spec['split'], spec['dataset_kind'], camera_name)
            except Exception as exc:
                skipped_cameras.append({'camera_name': camera_name, 'reason': str(exc)})
                continue

            pair_groups = _group_samples_by_pair(ctx)
            if args.max_pairs_per_scenario is not None:
                remaining = args.max_pairs_per_scenario - scenario_pair_count
                if remaining <= 0:
                    break
                pair_groups = pair_groups[:remaining]
            for seq_name, cam_name, items in tqdm(pair_groups, desc=f'{scenario_name}:{camera_name}:pairs', leave=False):
                if int(args.frame_subsample_stride) > 1:
                    items = items[:: int(args.frame_subsample_stride)]
                if args.max_frames_per_pair is not None:
                    items = items[: args.max_frames_per_pair]
                if not items:
                    continue
                if spec['dataset_kind'] == 'humman':
                    if humman_converter is None:
                        raise RuntimeError('HuMMan scenario requires the official MHR-to-SMPL converter, but it was not initialized.')
                    frame_results = _evaluate_humman_pair(
                        ctx=ctx,
                        items=items,
                        estimator=estimator,
                        converter=humman_converter,
                        use_mask=args.use_mask,
                        invalid_frame_mode=args.invalid_frame_mode,
                    )
                else:
                    frame_results = []
                    for sample_index, frame_id in items:
                        frame_results.append(
                            _evaluate_one_frame(
                                ctx=ctx,
                                sample_index=int(sample_index),
                                frame_id=int(frame_id),
                                estimator=estimator,
                                joint_adapter=panoptic_adapter,
                                use_mask=args.use_mask,
                                invalid_frame_mode=args.invalid_frame_mode,
                            )
                        )
                agg = _aggregate_frame_results(frame_results)
                pair_rows.append({
                    'scenario': scenario_name,
                    'dataset_kind': spec['dataset_kind'],
                    'sequence_name': seq_name,
                    'camera_name': cam_name,
                    'cfg_path': ctx.cfg_path,
                    **agg,
                })
                scenario_pair_count += 1

        if not pair_rows:
            raise RuntimeError(f'No evaluated pairs produced for scenario {scenario_name}. Skipped cameras: {skipped_cameras}')

        camera_groups: dict[str, list[dict]] = defaultdict(list)
        all_frame_rows = []
        for row in pair_rows:
            camera_groups[str(row['camera_name'])].append(row)
            all_frame_rows.extend(row['frame_metrics'])

        valid_rows = [row for row in all_frame_rows if bool(row.get('valid', False))]
        overall = {
            'scenario': scenario_name,
            'dataset_kind': spec['dataset_kind'],
            'num_pairs': int(len(pair_rows)),
            'num_total_frames': int(len(all_frame_rows)),
            'num_valid_frames': int(len(valid_rows)),
            'num_invalid_frames': int(len(all_frame_rows) - len(valid_rows)),
            'mpjpe_mean': float(np.mean([row['mpjpe'] for row in valid_rows])) if valid_rows else float('nan'),
            'pa_mpjpe_mean': float(np.mean([row['pa_mpjpe'] for row in valid_rows])) if valid_rows else float('nan'),
            'pc_mpjpe_mean': float(np.mean([row['pc_mpjpe'] for row in valid_rows])) if valid_rows else float('nan'),
            'mpjpe_max': float(np.max([row['mpjpe'] for row in valid_rows])) if valid_rows else float('nan'),
            'pa_mpjpe_max': float(np.max([row['pa_mpjpe'] for row in valid_rows])) if valid_rows else float('nan'),
            'pc_mpjpe_max': float(np.max([row['pc_mpjpe'] for row in valid_rows])) if valid_rows else float('nan'),
        }

        camera_rows = []
        for camera_name, rows in sorted(camera_groups.items()):
            frame_rows = []
            for row in rows:
                frame_rows.extend(row['frame_metrics'])
            valid = [x for x in frame_rows if bool(x.get('valid', False))]
            camera_rows.append({
                'scenario': scenario_name,
                'dataset_kind': spec['dataset_kind'],
                'camera_name': camera_name,
                'num_pairs': int(len(rows)),
                'num_total_frames': int(len(frame_rows)),
                'num_valid_frames': int(len(valid)),
                'num_invalid_frames': int(len(frame_rows) - len(valid)),
                'mpjpe_mean': float(np.mean([x['mpjpe'] for x in valid])) if valid else float('nan'),
                'pa_mpjpe_mean': float(np.mean([x['pa_mpjpe'] for x in valid])) if valid else float('nan'),
                'pc_mpjpe_mean': float(np.mean([x['pc_mpjpe'] for x in valid])) if valid else float('nan'),
                'mpjpe_max': float(np.max([x['mpjpe'] for x in valid])) if valid else float('nan'),
                'pa_mpjpe_max': float(np.max([x['pa_mpjpe'] for x in valid])) if valid else float('nan'),
                'pc_mpjpe_max': float(np.max([x['pc_mpjpe'] for x in valid])) if valid else float('nan'),
            })

        _write_csv(scenario_dir / 'pair_metrics.csv', pair_rows, pair_fieldnames)
        dump_json(scenario_dir / 'pair_metrics.json', pair_rows)
        _write_csv(scenario_dir / 'camera_summary.csv', camera_rows, camera_fieldnames)
        dump_json(scenario_dir / 'camera_summary.json', camera_rows)
        _write_csv(scenario_dir / 'overall_metrics.csv', [overall], overall_fieldnames)
        dump_json(scenario_dir / 'overall_metrics.json', overall)
        dump_json(scenario_dir / 'skipped_cameras.json', skipped_cameras)
        suite_summary.append(overall)

    _write_csv(run_dir / 'suite_summary.csv', suite_summary, overall_fieldnames)
    dump_json(run_dir / 'suite_summary.json', suite_summary)
    print(f'[sam3d-eval-suite] completed {len(suite_summary)} scenario(s). outputs: {run_dir}')


if __name__ == '__main__':
    main()
