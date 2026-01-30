import numpy as np


class AttachGTPoseInputs:
    def __init__(
        self,
        use_rgb: bool = True,
        use_depth: bool = True,
        rgb_key: str = "input_pose2d_rgb",
        depth_key: str = "input_pose3d_depth",
        drop_raw_inputs: bool = False,
    ):
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self.drop_raw_inputs = drop_raw_inputs

    def _get_seq_keypoints(self, sample):
        if "gt_keypoints_seq" in sample:
            return sample["gt_keypoints_seq"]
        if "gt_keypoints" in sample:
            keypoints = sample["gt_keypoints"]
            if keypoints is None:
                return None
            if "input_rgb" in sample:
                t = len(sample["input_rgb"])
            elif "input_depth" in sample:
                t = len(sample["input_depth"])
            else:
                t = 1
            keypoints = np.asarray(keypoints, dtype=np.float32)
            return np.repeat(keypoints[None, ...], t, axis=0)
        return None

    @staticmethod
    def _to_camera_frame(keypoints, camera):
        if camera is None or "extrinsic" not in camera:
            return keypoints
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float32)
        R = extrinsic[:, :3]
        T = extrinsic[:, 3]
        return (R @ keypoints.T).T + T

    def __call__(self, sample):
        keypoints_seq = self._get_seq_keypoints(sample)
        if keypoints_seq is None:
            return sample

        if self.use_rgb:
            rgb_cam = sample.get("rgb_camera")
            rgb_pose2d = []
            for kp in keypoints_seq:
                kp_cam = self._to_camera_frame(kp, rgb_cam)
                rgb_pose2d.append(kp_cam[:, :2].astype(np.float32))
            sample[self.rgb_key] = rgb_pose2d

        if self.use_depth:
            depth_cam = sample.get("depth_camera")
            depth_pose3d = []
            for kp in keypoints_seq:
                kp_cam = self._to_camera_frame(kp, depth_cam)
                depth_pose3d.append(kp_cam.astype(np.float32))
            sample[self.depth_key] = depth_pose3d

        if self.drop_raw_inputs:
            if "input_rgb" in sample:
                sample.pop("input_rgb")
            if "input_depth" in sample:
                sample.pop("input_depth")

        return sample
