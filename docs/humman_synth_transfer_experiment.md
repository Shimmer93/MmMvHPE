# HuMMan Synthetic Transfer Experiment

This workflow evaluates the HuMMan-style synthetic pretrained two-stage model, finetunes it on the real HuMMan dataset, evaluates the finetuned two-stage model again with the fixed-lidar-frame metric script, and writes a local report.

The runner is:
- `scripts/run_humman_synth_transfer_experiment.py`

It performs:
1. fixed-lidar-frame evaluation of the synthetic pretrained HuMMan stage-1 + stage-2 checkpoints
2. real-data HuMMan stage-1 finetuning from the synthetic stage-1 checkpoint
3. real-data HuMMan stage-2 finetuning from the real stage-1 checkpoint, warm-started from the synthetic stage-2 camera head
4. fixed-lidar-frame evaluation of the finetuned HuMMan stage-1 + stage-2 checkpoints
5. report generation under `logs/experiments/humman_synth_transfer/<run_id>/`
6. optional Notion page creation when `NOTION_TOKEN` and `NOTION_PARENT_PAGE_ID` are available in the environment

## Example

```bash
uv run python scripts/run_humman_synth_transfer_experiment.py \
  --run-id 20260320_humman_transfer \
  --gpus 2 \
  --num-workers 8 \
  --batch-size 32 \
  --batch-size-eval 32 \
  --pin-memory
```

## Outputs

The runner writes:
- per-step command logs
- temporary configs with injected checkpoint paths
- fixed-lidar-frame evaluation logs
- `summary.json`
- `report.md`
- `notion_response.json`

All outputs are stored under:

```text
logs/experiments/humman_synth_transfer/<run_id>/
```
