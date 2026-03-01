from typing import List, Optional, Sequence

from datasets.humman_dataset_v2 import HummanPreprocessedDatasetV2


def build_depth_to_lidar_pipeline(num_points: int = 1024) -> List[dict]:
    num_points = int(num_points)
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}.")
    return [
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
                "num_points": num_points,
                "pad_mode": "repeat",
                "keys": ["input_lidar"],
            },
        },
        {"name": "ToTensor", "params": None},
    ]


class HummanDepthToLidarDataset(HummanPreprocessedDatasetV2):
    """Depth-only HuMMan dataset that uses the V2 depth->LiDAR conversion path."""

    def __init__(
        self,
        data_root: str,
        pipeline: Optional[List[dict]] = None,
        split: str = "train",
        split_config: Optional[str] = None,
        split_to_use: str = "random_split",
        unit: str = "m",
        depth_cameras: Optional[Sequence[str]] = None,
        seq_len: int = 1,
        seq_step: int = 1,
        pad_seq: bool = True,
        causal: bool = True,
        use_all_pairs: bool = False,
        test_mode: bool = False,
        apply_to_new_world: bool = True,
        remove_root_rotation: bool = True,
        colocated: bool = False,
    ) -> None:
        if pipeline is None:
            pipeline = build_depth_to_lidar_pipeline(num_points=1024)

        super().__init__(
            data_root=data_root,
            unit=unit,
            pipeline=pipeline,
            split=split,
            split_config=split_config,
            split_to_use=split_to_use,
            test_mode=test_mode,
            modality_names=("depth",),
            rgb_cameras=(),
            depth_cameras=depth_cameras,
            rgb_cameras_per_sample=1,
            depth_cameras_per_sample=1,
            lidar_cameras_per_sample=1,
            seq_len=seq_len,
            seq_step=seq_step,
            pad_seq=pad_seq,
            causal=causal,
            use_all_pairs=use_all_pairs,
            colocated=colocated,
            convert_depth_to_lidar=True,
            apply_to_new_world=apply_to_new_world,
            remove_root_rotation=remove_root_rotation,
            skeleton_only=True,
        )
