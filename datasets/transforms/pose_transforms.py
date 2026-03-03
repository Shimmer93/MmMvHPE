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


class AttachGTPose2DH36M:
    def __init__(
        self,
        input_key: str = "gt_keypoints_by_view",
        camera_key: str = "rgb_cameras",
        output_key: str = "input_pose2d_rgb",
        clamp_min_z: float = 1e-4,
    ):
        self.input_key = input_key
        self.camera_key = camera_key
        self.output_key = output_key
        self.clamp_min_z = clamp_min_z

    def __call__(self, sample):
        keypoints = sample.get(self.input_key)
        cameras = sample.get(self.camera_key)
        if keypoints is None or cameras is None:
            return sample

        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.ndim != 4:
            raise ValueError(f"Expected keypoints shape [V, T, J, 3], got {keypoints.shape}")

        V, T, J, _ = keypoints.shape
        if len(cameras) != V:
            raise ValueError(f"Camera count {len(cameras)} does not match views {V}")

        pose2d = []
        for v in range(V):
            cam = cameras[v]
            K = cam["intrinsic"]
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            view_pose2d = []
            for t in range(T):
                pts = keypoints[v, t]
                Z = np.clip(pts[:, 2], self.clamp_min_z, None)
                x = fx * (pts[:, 0] / Z) + cx
                y = fy * (pts[:, 1] / Z) + cy
                view_pose2d.append(np.stack([x, y], axis=-1))
            pose2d.append(np.stack(view_pose2d, axis=0))

        sample[self.output_key] = np.stack(pose2d, axis=0)
        return sample
