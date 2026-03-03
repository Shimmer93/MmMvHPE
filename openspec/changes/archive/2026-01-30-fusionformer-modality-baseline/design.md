## Context

The repo currently trains multimodal models on HummanPreprocessedDatasetV2, which yields one RGB view and one depth/point cloud view per sample. FusionFormer (AAAI-24) is a camera-parameter-free Transformer that fuses multi-view and multi-frame pose tokens. For this baseline, “views” are reinterpreted as modalities (RGB vs depth/PC), and GT 2D/3D poses are used as stand-ins for off-the-shelf estimators. The model outputs 3D keypoints only. CUDA/GPU is expected for the Transformer blocks and any heavy tensor ops; preprocessing remains lightweight.

## Goals / Non-Goals

**Goals:**
- Implement a FusionFormer-style network that fuses modality tokens over time and predicts 3D keypoints.
- Provide GT pose adapters that emit 2D (RGB) and 3D (PC) pose inputs consistent with the baseline.
- Add a minimal train_mini/test_mini config to validate the training/eval loop on HummanPreprocessedDatasetV2.

**Non-Goals:**
- Training or integrating real 2D/3D estimators (kept as future work).
- True multi-view fusion within a modality (no multi-camera stacking).
- SMPL regression outputs or camera-parameter usage.

## Decisions

- **Modality-as-view tokens:** Treat modalities as “views” and fuse tokens from RGB-2D and PC-3D pose sources in a single unified Transformer. This aligns with the paper’s fusion pattern while matching current dataset output.
- **GT pose adapters:** Provide lightweight adapters that emit 2D/3D joint tensors without external inference; this keeps the baseline reproducible and avoids adding estimator dependencies.
- **Unified fusion structure:** Implement the paper’s encoder-decoder fusion with learnable positional encoding, but collapse to a single global head to output one 3D pose (not per-modality) since the target is a single keypoint set.
- **Config-driven cameras:** Use all available cameras by default via config, but keep the dataset as-is to avoid refactoring for true multi-view.
- **Compute placement:** Transformer encoder/decoder and temporal aggregation live in `models/` and run on GPU; GT pose adaptation remains in dataset/pipeline and stays CPU.

## Risks / Trade-offs

- [Deviation from paper multi-view] → Mitigate by documenting “modality-as-view” assumption and keeping interfaces compatible with future multi-view expansion.
- [GT pose adapters may mask estimator error] → Mitigate by treating this as a baseline and planning a follow-up with real estimators.
- [Potential mismatch in joint definitions] → Mitigate by enforcing a single joint order (Humman/SMPL 24 joints) and validating with MPJPE on GT.
- [Single-view dataset limits fusion benefits] → Mitigate by focusing on verifying architecture integration and correctness rather than SOTA performance.
