import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os
import builtins

# -----------------------------------------------------------------------------
# [New Feature] Custom Print to filter [DEBUG] messages
# -----------------------------------------------------------------------------
_original_print = builtins.print

def _debug_aware_print(*args, **kwargs):
    # Convert args to a single string to check for the tag
    msg = " ".join([str(arg) for arg in args])
    
    # If the message is a debug message
    if "[DEBUG]" in msg:
        # Only print if the environment variable is set to '1'
        if os.environ.get('MmMvHPE_DEBUG', '0') == '1':
            _original_print(*args, **kwargs)
    else:
        # Always print non-debug messages
        _original_print(*args, **kwargs)

# Override the built-in print function
builtins.print = _debug_aware_print
# -----------------------------------------------------------------------------

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
import wandb

from datasets.data_api import LitDataModule
from models.model_api import LitModel
from misc.utils import load_cfg, merge_args_cfg

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

def main(args):
    if hasattr(args, 'seed'):
        L.seed_everything(args.seed, workers=True)
    print("[DEBUG]: The arguments has the attributes input_dim or not:", hasattr(args, 'input_dim'))

    dm = LitDataModule(hparams=args)
    model = LitModel(hparams=args)
    
    monitor = 'val_mpjpe'

    if hasattr(args, 'epochs'):
        filename = args.model_name+'-{epoch}-{'+monitor+':.4f}'
    else:
        filename = args.model_name+'-{step}-{'+monitor+':.4f}'

    callbacks = [
        ModelCheckpoint(
            monitor=monitor,
            dirpath=os.path.join('logs', args.exp_name, args.version),
            filename=filename,
            save_top_k=1,
            save_last=True,
            mode='min'),
        RichProgressBar(refresh_rate=20),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # print("[DEBUG]: this is the line 47 of main.py")

    if args.use_wandb:
        logger = WandbLogger(
            project=args.exp_name,
            name=args.version,
            save_dir='logs',
            offline=args.wandb_offline,
            log_model=False,
            job_type='test' if args.test or args.predict else 'train'
        )
    else:
        logger = TensorBoardLogger(
            save_dir='logs', 
            name=args.exp_name,
            version=args.version,
            default_hp_metric=False
        )
    logger.log_hyperparams(args)

    trainer_kwargs = {
        'fast_dev_run': args.dev,
        'logger': logger,
        'devices': 1 if args.predict else args.gpus,
        'accelerator': "gpu",
        'sync_batchnorm': args.sync_batchnorm,
        'num_nodes': args.num_nodes,
        'gradient_clip_val': args.clip_grad,
        'strategy': DDPStrategy(
            find_unused_parameters=False,  # Set to False for better performance
            gradient_as_bucket_view=True,  # More efficient gradient sync
            static_graph=True  # If your model structure doesn't change
        ) if args.strategy == 'ddp' else args.strategy,
        'callbacks': callbacks,
        'precision': args.precision,
        'benchmark': args.benchmark,  # cudnn benchmark
        'deterministic': args.deterministic,  # Set True for reproducibility (slower)
    }
    # print("[DEBUG]: this is the line 83 of main.py")


    if hasattr(args, 'epochs'):
        print(f'Training for {args.epochs} epochs.')
        trainer_kwargs.update({
            'max_epochs': args.epochs,
        })
    else:
        # step-based training
        print(f'Training for {args.max_steps} steps.')
        trainer_kwargs.update({
            'max_epochs': -1,
            'max_steps': args.max_steps,
            'val_check_interval': args.val_check_interval,
            'limit_val_batches': args.limit_val_batches,
        })

    trainer = L.Trainer(**trainer_kwargs)

    if bool(args.test):
        trainer.test(model, datamodule=dm, ckpt_path=args.checkpoint_path)
    elif bool(args.predict):
        predictions = trainer.predict(
            model, 
            datamodule=dm, 
            return_predictions=True,
            ckpt_path=args.checkpoint_path
        )
        save_path = os.path.join('logs', args.exp_name, args.version, f'{args.model_name}_predictions.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f'Saving predictions to {save_path}')
        torch.save(predictions, save_path)
    else:
        trainer.fit(
            model, 
            datamodule=dm,
            ckpt_path=args.checkpoint_path if hasattr(args, 'resume') and args.resume else None
        )
        if args.dev == 0:
            trainer.test(ckpt_path="best", datamodule=dm)

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/test.yaml')
    parser.add_argument('-g', '--gpus', type=int, default=None,
                        help='number of gpus to use (default: all available)')
    parser.add_argument('-d', "--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument('-n', "--num_nodes", type=int, default=1)
    parser.add_argument('-w', "--num_workers", type=int, default=4)
    parser.add_argument('-b', "--batch_size", type=int, default=2048)
    parser.add_argument('-e', "--batch_size_eva", type=int, default=1000, help='batch_size for evaluation')
    parser.add_argument('-p', "--prefetch_factor", type=int, default=2, help='DataLoader prefetch factor')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_test_preds', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--exp_name', type=str, default='fasternet')
    parser.add_argument("--version", type=str, default="0")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logger')
    parser.add_argument('--wandb_offline', action='store_true', help='Run wandb in offline mode')
    parser.add_argument('--deterministic', action='store_true', help='Set cudnn.deterministic=True for reproducibility')
    
    # [New Argument]
    parser.add_argument('--debug', action='store_true', help='Enable printing of [DEBUG] messages')

    args = parser.parse_args()
    
    # [New Feature] Set environment variable based on argument
    os.environ['MmMvHPE_DEBUG'] = '1' if args.debug else '0'
    
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    main(args)