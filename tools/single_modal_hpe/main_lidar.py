import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.single_model_hpe.dataset import HummanDepthToLidarDataset
from tools.single_model_hpe.model import SimpleLidarHPEModel
from tools.single_model_hpe.train_eval import (
    evaluate,
    export_predictions_as_mmpose_json,
    load_checkpoint,
    save_checkpoint,
    test_and_save_predictions,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test a simple LiDAR-only HPE model.")
    parser.add_argument("--data-root", type=str, required=True, help="Preprocessed HuMMan root path.")
    parser.add_argument(
        "--split-config",
        type=str,
        default="configs/datasets/humman_split_config.yml",
        help="Optional split config yaml.",
    )
    parser.add_argument("--split-to-use", type=str, default="cross_camera_split")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--depth-cameras", nargs="+", default=None)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--seq-step", type=int, default=1)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--unit", type=str, default="m", choices=["m", "mm"])
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--min-depth", type=float, default=1e-6)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)

    parser.add_argument("--num-joints", type=int, default=24)
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=1024)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--spatial-stride", type=int, default=32)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--depth-mamba-inter", type=int, default=5)
    parser.add_argument("--depth-mamba-intra", type=int, default=1)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--mode", type=str, default="xyz", choices=["xyz", "d", "all", "only_h"])

    parser.add_argument("--output-dir", type=str, default="logs/single_model_hpe")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--pred-file", type=str, default="test_predictions.npz")
    parser.add_argument(
        "--export-all-json",
        type=str,
        default=None,
        help="If set, export frame-wise predictions for the entire dataset in MMPose JSON format.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, drop_last: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_model(args: argparse.Namespace) -> SimpleLidarHPEModel:
    encoder_kwargs = {
        "radius": args.radius,
        "nsamples": args.nsamples,
        "spatial_stride": args.spatial_stride,
        "temporal_kernel_size": args.temporal_kernel_size,
        "temporal_stride": args.temporal_stride,
        "dim": args.encoder_dim,
        "mlp_dim": args.encoder_dim * 2,
        "depth_mamba_inter": args.depth_mamba_inter,
        "depth_mamba_intra": args.depth_mamba_intra,
        "drop_path": args.drop_path,
        "mode": args.mode,
    }
    return SimpleLidarHPEModel(
        num_joints=args.num_joints,
        encoder_dim=args.encoder_dim,
        head_hidden_dim=args.head_hidden_dim,
        encoder_kwargs=encoder_kwargs,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_start_time = time.time()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but no CUDA GPU is available.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("MAMBA4DEncoder requires CUDA. Use --device cuda.")

    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    split_config = args.split_config if args.split_config and os.path.isfile(args.split_config) else None
    if args.split_config and split_config is None:
        print(f"[WARN] split config not found at {args.split_config}; falling back to internal split.")

    # Match the requested training pipeline.
    common_pipeline = [
        {
            "name": "CameraParamToPoseEncoding",
            "params": {"pose_encoding_type": "absT_quaR_FoV"},
        },
        {
            "name": "PCCenterWithKeypoints",
            "params": {
                "center_type": "mean",
                "keys": ["input_lidar"],
                "keypoints_key": "gt_keypoints",
            },
        },
        {
            "name": "PCPad",
            "params": {
                "num_points": 1024,
                "pad_mode": "repeat",
                "keys": ["input_lidar"],
            },
        },
        {"name": "ToTensor", "params": None},
    ]

    train_dataset = HummanDepthToLidarDataset(
        data_root=args.data_root,
        pipeline=common_pipeline,
        split=args.train_split,
        split_config=split_config,
        split_to_use=args.split_to_use,
        unit=args.unit,
        depth_cameras=args.depth_cameras,
        seq_len=args.seq_len,
        seq_step=args.seq_step,
        causal=args.causal,
        test_mode=False,
        num_points=args.num_points,
        min_depth=args.min_depth,
    )
    test_dataset = HummanDepthToLidarDataset(
        data_root=args.data_root,
        pipeline=common_pipeline,
        split=args.test_split,
        split_config=split_config,
        split_to_use=args.split_to_use,
        unit=args.unit,
        depth_cameras=args.depth_cameras,
        seq_len=args.seq_len,
        seq_step=args.seq_step,
        causal=args.causal,
        test_mode=True,
        num_points=args.num_points,
        min_depth=args.min_depth,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_ckpt_path = os.path.join(args.output_dir, "best.pt")
    last_ckpt_path = os.path.join(args.output_dir, "last.pt")
    pred_path = os.path.join(args.output_dir, args.pred_file)
    export_json_path = None
    if args.export_all_json:
        export_json_path = args.export_all_json
        if not os.path.isabs(export_json_path):
            export_json_path = os.path.join(args.output_dir, export_json_path)

    if args.test_only:
        ckpt = args.checkpoint if args.checkpoint is not None else best_ckpt_path
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        load_checkpoint(ckpt, model, map_location="cpu")
        test_metrics = test_and_save_predictions(
            model,
            test_loader,
            device,
            pred_path,
            log_interval=args.log_interval,
        )
        print(f"Test MPJPE: {test_metrics['mpjpe']:.6f}")
        print(f"Test elapsed: {test_metrics['elapsed_sec']:.1f}s")
        print(f"Predictions saved to: {pred_path}")
        if export_json_path is not None:
            export_dataset = HummanDepthToLidarDataset(
                data_root=args.data_root,
                pipeline=common_pipeline,
                split="all",
                split_config=None,
                split_to_use=args.split_to_use,
                unit=args.unit,
                depth_cameras=None,
                seq_len=1,
                seq_step=1,
                causal=True,
                test_mode=True,
                num_points=args.num_points,
                min_depth=args.min_depth,
            )
            export_loader = build_dataloader(
                export_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
            export_metrics = export_predictions_as_mmpose_json(
                model=model,
                dataloader=export_loader,
                device=device,
                save_path=export_json_path,
                input_dir=os.path.join(args.data_root, "depth"),
                config="tools/single_model_hpe/main_lidar.py",
                checkpoint=ckpt,
                device_str=args.device,
                log_interval=args.log_interval,
            )
            print(
                f"Exported full-dataset JSON: {export_json_path} "
                f"(num_images={int(export_metrics['num_images'])}, "
                f"elapsed={export_metrics['elapsed_sec']:.1f}s)"
            )
        print(f"Total elapsed: {time.time() - run_start_time:.1f}s")
        return

    best_val_mpjpe = float("inf")
    start_epoch = 0

    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        loaded_epoch, best_val_mpjpe = load_checkpoint(
            args.checkpoint,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
        )
        start_epoch = loaded_epoch + 1
        print(f"Resumed from {args.checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            epoch_idx=epoch,
            num_epochs=args.epochs,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            log_interval=args.log_interval,
            epoch_idx=epoch,
            num_epochs=args.epochs,
        )
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_mpjpe={train_metrics['mpjpe']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mpjpe={val_metrics['mpjpe']:.6f} "
            f"train_elapsed={train_metrics['elapsed_sec']:.1f}s "
            f"val_elapsed={val_metrics['elapsed_sec']:.1f}s "
            f"total_elapsed={time.time() - run_start_time:.1f}s"
        )

        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, best_val_mpjpe)
        if val_metrics["mpjpe"] < best_val_mpjpe:
            best_val_mpjpe = val_metrics["mpjpe"]
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_val_mpjpe)
            print(f"Saved best checkpoint to {best_ckpt_path}")

    if not os.path.isfile(best_ckpt_path):
        raise RuntimeError("Best checkpoint was not created.")

    load_checkpoint(best_ckpt_path, model, map_location="cpu")
    test_metrics = test_and_save_predictions(
        model,
        test_loader,
        device,
        pred_path,
        log_interval=args.log_interval,
    )
    print(f"Final Test MPJPE: {test_metrics['mpjpe']:.6f}")
    print(f"Test elapsed: {test_metrics['elapsed_sec']:.1f}s")
    print(f"Predictions saved to: {pred_path}")
    if export_json_path is not None:
        export_dataset = HummanDepthToLidarDataset(
            data_root=args.data_root,
            pipeline=common_pipeline,
            split="all",
            split_config=None,
            split_to_use=args.split_to_use,
            unit=args.unit,
            depth_cameras=None,
            seq_len=1,
            seq_step=1,
            causal=True,
            test_mode=True,
            num_points=args.num_points,
            min_depth=args.min_depth,
        )
        export_loader = build_dataloader(
            export_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        export_metrics = export_predictions_as_mmpose_json(
            model=model,
            dataloader=export_loader,
            device=device,
            save_path=export_json_path,
            input_dir=os.path.join(args.data_root, "depth"),
            config="tools/single_model_hpe/main_lidar.py",
            checkpoint=best_ckpt_path,
            device_str=args.device,
            log_interval=args.log_interval,
        )
        print(
            f"Exported full-dataset JSON: {export_json_path} "
            f"(num_images={int(export_metrics['num_images'])}, "
            f"elapsed={export_metrics['elapsed_sec']:.1f}s)"
        )
    print(f"Total elapsed: {time.time() - run_start_time:.1f}s")


if __name__ == "__main__":
    main()
