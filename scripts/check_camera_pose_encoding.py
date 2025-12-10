import argparse
import pprint

from datasets.h36m_dataset import H36MDataset


def build_dataset(args):
    # Minimal pipeline that only generates camera pose encodings.
    pipeline = [
        {"name": "CameraParamToPoseEncoding", "params": {"pose_encoding_type": "absT_quaR_FoV"}},
    ]

    dataset = H36MDataset(
        data_root=args.data_root,
        split=args.split,
        modality_names=args.modalities,
        cameras=args.cameras,
        seq_len=args.seq_len,
        seq_step=1,
        pad_seq=True,
        causal=True,
        pipeline=pipeline,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Quick check for camera pose encoding transform.")
    parser.add_argument("--data_root", type=str, default="/data/shared/H36M-Toolbox")
    parser.add_argument("--split", type=str, default="train_mini")
    parser.add_argument("--modalities", nargs="+", default=["rgb", "depth"])
    parser.add_argument("--cameras", nargs="+", default=["02"])
    parser.add_argument("--seq_len", type=int, default=1)
    args = parser.parse_args()

    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty with the given configuration.")

    sample = dataset[0]

    print("Sample keys:", list(sample.keys()))
    for modality in args.modalities:
        key = f"gt_camera_{modality}"
        assert key in sample, f"Missing pose encoding for modality '{modality}'."
        pose_enc = sample[key]
        print(f"{key} shape: {pose_enc.shape}")

    # Show a compact preview of the camera encodings (first element).
    pprint.pp(
        {f"gt_camera_{m}": sample[f"gt_camera_{m}"][0].tolist() for m in args.modalities}
    )
    print("Camera pose encoding transform looks valid.")


if __name__ == "__main__":
    main()

