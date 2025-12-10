import numpy as np
import torch

from misc.pose_enc import extri_intri_to_pose_encoding


class CameraParamToPoseEncoding:
    """Convert camera intrinsics/extrinsics into pose encodings for supervision."""

    def __init__(self, pose_encoding_type: str = "absT_quaR_FoV"):
        self.pose_encoding_type = pose_encoding_type

    def __call__(self, sample):
        modalities = sample.get("modalities", [])

        for modality in modalities:
            camera_key = f"{modality}_camera"
            input_key = f"input_{modality}"

            camera = sample.get(camera_key)
            frames = sample.get(input_key)
            if camera is None or frames is None:
                continue

            # Prepare batched extrinsics/intrinsics for the sequence length
            seq_len = len(frames)
            extrinsic = np.asarray(camera["extrinsic"], dtype=np.float32)
            intrinsic = np.asarray(camera["intrinsic"], dtype=np.float32)

            extrinsics = torch.from_numpy(np.stack([extrinsic] * seq_len, axis=0)).unsqueeze(0)
            intrinsics = torch.from_numpy(np.stack([intrinsic] * seq_len, axis=0)).unsqueeze(0)

            # Image size is (H, W)
            height, width = frames[0].shape[:2]
            pose_enc = extri_intri_to_pose_encoding(
                extrinsics,
                intrinsics,
                image_size_hw=(height, width),
                pose_encoding_type=self.pose_encoding_type,
            )

            # Store per-sample (S x 9) pose encoding; collate will add batch dim
            sample[f"gt_camera_{modality}"] = pose_enc.squeeze(0)

        return sample
