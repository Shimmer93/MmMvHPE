import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.single_modal_hpe.dataset import HummanDepthToLidarDataset, build_depth_to_lidar_pipeline
from tools.single_modal_hpe.train_eval import (
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
    parser.add_argument("--split-to-use", type=str, default="random_split")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--depth-cameras", nargs="+", default=None)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--seq-step", type=int, default=1)
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use causal target selection (matches main config default).",
    )
    parser.add_argument("--unit", type=str, default="m", choices=["m", "mm"])
    parser.add_argument("--num-points", type=int, default=1024)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--log-overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use in-place progress logs instead of printing a new line each update.",
    )

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

    parser.add_argument("--output-dir", type=str, default="logs/single_modal_hpe")
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
    parser.add_argument(
        "--strategy",
        type=str,
        default="single",
        choices=["single", "dp", "ddp"],
        help="Single GPU, DataParallel, or DistributedDataParallel.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids for DP (e.g., 0,1,2). Defaults to all visible GPUs.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    sampler=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_model(args: argparse.Namespace):
    from tools.single_modal_hpe.model import SimpleLidarHPEModel

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


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _init_ddp() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ or "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "DDP requires torchrun environment variables LOCAL_RANK/RANK/WORLD_SIZE. "
            "Launch with: torchrun --nproc_per_node=<num_gpus> ..."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("DDP strategy selected but WORLD_SIZE < 2.")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def _cleanup_ddp() -> None:
    if _dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _parse_gpu_ids(gpu_ids_str: str | None) -> list[int]:
    if gpu_ids_str is None:
        return list(range(torch.cuda.device_count()))
    out = []
    for tok in gpu_ids_str.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    if len(out) == 0:
        raise ValueError("`--gpu-ids` is empty after parsing.")
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_start_time = time.time()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but no CUDA GPU is available.")

    if args.strategy == "ddp":
        local_rank, rank, world_size = _init_ddp()
        device = torch.device("cuda", local_rank)
        is_main = _is_main_process(rank)
    else:
        local_rank, rank, world_size = 0, 0, 1
        is_main = True
        device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("MAMBA4DEncoder requires CUDA. Use --device cuda.")

    torch.backends.cudnn.benchmark = True
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    split_config = args.split_config if args.split_config and os.path.isfile(args.split_config) else None
    if args.split_config and split_config is None and is_main:
        print(f"[WARN] split config not found at {args.split_config}; falling back to internal split.")

    common_pipeline = build_depth_to_lidar_pipeline(num_points=args.num_points)

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
        pad_seq=True,
        causal=args.causal,
        use_all_pairs=False,
        test_mode=False,
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
        pad_seq=True,
        causal=args.causal,
        use_all_pairs=False,
        test_mode=True,
    )

    if is_main:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples:  {len(test_dataset)}")

    pin_memory = device.type == "cuda"
    train_sampler = None
    test_sampler = None
    if args.strategy == "ddp":
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        sampler=train_sampler,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        sampler=test_sampler,
    )
    test_loader_full = None
    if args.strategy == "ddp" and is_main:
        test_loader_full = build_dataloader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            sampler=None,
        )

    model = build_model(args).to(device)
    if args.strategy == "dp":
        gpu_ids = _parse_gpu_ids(args.gpu_ids)
        if len(gpu_ids) < 2:
            raise ValueError("DP strategy requires at least 2 GPU ids.")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    elif args.strategy == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

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
        if args.strategy == "ddp" and not is_main:
            _cleanup_ddp()
            return
        ckpt = args.checkpoint if args.checkpoint is not None else best_ckpt_path
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        load_checkpoint(ckpt, _unwrap_model(model), map_location="cpu")
        test_metrics = test_and_save_predictions(
            model,
            test_loader_full if test_loader_full is not None else test_loader,
            device,
            pred_path,
            log_interval=args.log_interval if is_main else 0,
            log_overwrite=args.log_overwrite and is_main,
            sync_dist=False,
        )
        if is_main:
            print(f"Test MPJPE (centered): {test_metrics['mpjpe_centered']:.6f}")
            print(f"Test MPJPE (restored): {test_metrics['mpjpe_restored']:.6f}")
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
                pad_seq=True,
                causal=True,
                use_all_pairs=False,
                test_mode=True,
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
                config="tools/single_modal_hpe/main_lidar.py",
                checkpoint=ckpt,
                device_str=args.device,
                log_interval=args.log_interval,
                log_overwrite=args.log_overwrite,
            )
            if is_main:
                print(
                    f"Exported full-dataset JSON: {export_json_path} "
                    f"(num_images={int(export_metrics['num_images'])}, "
                    f"elapsed={export_metrics['elapsed_sec']:.1f}s)"
                )
        if is_main:
            print(f"Total elapsed: {time.time() - run_start_time:.1f}s")
        _cleanup_ddp()
        return

    best_val_mpjpe = float("inf")
    start_epoch = 0

    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        loaded_epoch, best_val_mpjpe = load_checkpoint(
            args.checkpoint,
            _unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
        )
        start_epoch = loaded_epoch + 1
        if is_main:
            print(f"Resumed from {args.checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval if is_main else 0,
            log_overwrite=args.log_overwrite and is_main,
            epoch_idx=epoch,
            num_epochs=args.epochs,
            sync_dist=(args.strategy == "ddp"),
        )
        val_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            log_interval=args.log_interval if is_main else 0,
            log_overwrite=args.log_overwrite and is_main,
            epoch_idx=epoch,
            num_epochs=args.epochs,
            sync_dist=(args.strategy == "ddp"),
        )
        scheduler.step()

        current_val_mpjpe = val_metrics["mpjpe_restored"]
        if not np.isfinite(current_val_mpjpe):
            current_val_mpjpe = val_metrics["mpjpe_centered"]

        if is_main:
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_mpjpe_centered={train_metrics['mpjpe_centered']:.6f} "
                f"train_mpjpe_restored={train_metrics['mpjpe_restored']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_mpjpe_centered={val_metrics['mpjpe_centered']:.6f} "
                f"val_mpjpe_restored={val_metrics['mpjpe_restored']:.6f} "
                f"train_elapsed={train_metrics['elapsed_sec']:.1f}s "
                f"val_elapsed={val_metrics['elapsed_sec']:.1f}s "
                f"total_elapsed={time.time() - run_start_time:.1f}s"
            )

            save_checkpoint(last_ckpt_path, _unwrap_model(model), optimizer, scheduler, epoch, best_val_mpjpe)
            if current_val_mpjpe < best_val_mpjpe:
                best_val_mpjpe = current_val_mpjpe
                save_checkpoint(best_ckpt_path, _unwrap_model(model), optimizer, scheduler, epoch, best_val_mpjpe)
                print(f"Saved best checkpoint to {best_ckpt_path}")
        if args.strategy == "ddp":
            dist.barrier()

    if is_main and not os.path.isfile(best_ckpt_path):
        raise RuntimeError("Best checkpoint was not created.")

    if args.strategy == "ddp":
        dist.barrier()

    if is_main:
        load_checkpoint(best_ckpt_path, _unwrap_model(model), map_location="cpu")
        test_metrics = test_and_save_predictions(
            model,
            test_loader_full if test_loader_full is not None else test_loader,
            device,
            pred_path,
            log_interval=args.log_interval,
            log_overwrite=args.log_overwrite,
            sync_dist=False,
        )
        print(f"Final Test MPJPE (centered): {test_metrics['mpjpe_centered']:.6f}")
        print(f"Final Test MPJPE (restored): {test_metrics['mpjpe_restored']:.6f}")
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
                pad_seq=True,
                causal=True,
                use_all_pairs=False,
                test_mode=True,
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
                config="tools/single_modal_hpe/main_lidar.py",
                checkpoint=best_ckpt_path,
                device_str=args.device,
                log_interval=args.log_interval,
                log_overwrite=args.log_overwrite,
            )
            print(
                f"Exported full-dataset JSON: {export_json_path} "
                f"(num_images={int(export_metrics['num_images'])}, "
                f"elapsed={export_metrics['elapsed_sec']:.1f}s)"
            )
        print(f"Total elapsed: {time.time() - run_start_time:.1f}s")

    _cleanup_ddp()


if __name__ == "__main__":
    main()
