## 1. Sequence Head + Loss

- [x] 1.1 Implement a sequence regression keypoint head that outputs `B, T, J, 3` and uses uniform per‑frame MSE loss
- [x] 1.2 Add shape assertions to guard sequence dimensions

## 2. Mamba4D Integration

- [x] 2.1 Wire Mamba4D encoder outputs into the sequence head (no conv1d) for HuMMan point‑cloud input

## 3. Config + Verification

- [x] 3.1 Add HuMMan config that trains Mamba4D sequence estimator with point‑cloud only inputs
- [x] 3.2 Run a short `uv run` sanity check for the new config
