import os
import os.path as osp
import json
import re
from typing import List, Optional, Sequence

import numpy as np
import cv2
import torch


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def normalize_2d_kp(kp_2d: np.ndarray, crop_size: int = 224) -> np.ndarray:
    ratio = 1.0 / float(crop_size)
    return 2.0 * kp_2d * ratio - 1.0


def trans_point2d(pt_2d: np.ndarray, trans: np.ndarray) -> np.ndarray:
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.0], dtype=np.float32)
    dst_pt = trans @ src_pt
    return dst_pt[:2]


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180.0
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_downdir = np.array([0, src_h * 0.5], dtype=np.float32)
    src_rightdir = np.array([src_w * 0.5, 0], dtype=np.float32)
    src_downdir = np.array([src_downdir[0] * cs - src_downdir[1] * sn,
                            src_downdir[0] * sn + src_downdir[1] * cs], dtype=np.float32)
    src_rightdir = np.array([src_rightdir[0] * cs - src_rightdir[1] * sn,
                             src_rightdir[0] * sn + src_rightdir[1] * cs], dtype=np.float32)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = src_center
    src[1] = src_center + src_downdir
    src[2] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = dst_center
    dst[1] = dst_center + dst_downdir
    dst[2] = dst_center + dst_rightdir

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def transfrom_keypoints(kp_2d: np.ndarray, center_x: float, center_y: float,
                        width: float, height: float, patch_width: int,
                        patch_height: int, scale: float = 1.2) -> np.ndarray:
    trans = gen_trans_from_patch_cv(center_x, center_y, width, height,
                                    patch_width, patch_height, scale, rot=0.0, inv=False)
    for i in range(kp_2d.shape[0]):
        kp_2d[i] = trans_point2d(kp_2d[i], trans)
    return kp_2d


class LEIRDepthToLiDARPC:
    def __init__(self,
                 focal_length: Optional[float] = None,
                 center: Optional[Sequence[float]] = None,
                 keys: List[str] = ['input_depth'],
                 camera_key: str = 'depth_camera',
                 min_depth: float = 1e-6):
        self.focal_length = focal_length
        self.center = center
        self.keys = keys
        self.camera_key = camera_key
        self.min_depth = min_depth

    def __call__(self, results):
        for key in results.keys():
            assert 'lidar' not in key.lower(), f"Key '{key}' seems to already contain LiDAR data."

        for key in self.keys:
            depth_seq = results[key]
            pc_seq = []

            has_camera = self.camera_key in results
            if has_camera:
                camera = results[self.camera_key]
                K = np.array(camera['intrinsic'], dtype=np.float32)
                K_inv = np.linalg.inv(K)
                extrinsic = np.array(camera['extrinsic'], dtype=np.float32)
                R = extrinsic[:, :3]
                T = extrinsic[:, 3:].reshape(3, 1)
            else:
                if self.focal_length is None:
                    raise KeyError(
                        f"Missing camera parameters '{self.camera_key}' in results and no focal_length provided."
                    )

            for depth in depth_seq:
                H, W = depth.shape
                xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
                z = depth.reshape(-1)
                valid = z > self.min_depth

                if has_camera:
                    pixels = np.stack([xmap.reshape(-1), ymap.reshape(-1), np.ones(H * W)], axis=0)
                    rays = K_inv @ pixels
                    cam_points = rays * z
                    cam_points = cam_points[:, valid]
                    cam_points = (R.T @ (cam_points - T)).T
                    pc = cam_points.astype(np.float32)
                else:
                    if self.center is None:
                        cx, cy = W / 2.0, H / 2.0
                    else:
                        cx, cy = self.center
                    x = (xmap.reshape(-1) - cx) * z / self.focal_length
                    y = (ymap.reshape(-1) - cy) * z / self.focal_length
                    pc = np.vstack((x, y, z)).T
                    pc = pc[valid].astype(np.float32)
                pc_seq.append(pc)

            out_key = key.replace('depth', 'lidar')
            results[out_key] = pc_seq
            if 'modalities' in results and 'lidar' not in results['modalities']:
                results['modalities'].append('lidar')

        return results


class LEIRPrepareGT:
    def __init__(
        self,
        data_root: str,
        unit: str = "m",
        causal: bool = False,
        rgb_camera_key: str = "rgb_camera",
    ):
        self.data_root = data_root
        self.unit = unit
        self.causal = causal
        self.rgb_camera_key = rgb_camera_key
        self._seq_re = re.compile(r"^(p\d+_a\d+)_rgb_(kinect_\d{3}|iphone)_depth_(kinect_\d{3}|iphone)_(\d+)$")

    def _load_camera_params(self, seq_name):
        camera_file = osp.join(self.data_root, "cameras", f"{seq_name}_cameras.json")
        with open(camera_file, "r") as f:
            cameras = json.load(f)
        return cameras

    def _load_smpl_params(self, seq_name):
        smpl_file = osp.join(self.data_root, "smpl", f"{seq_name}_smpl_params.npz")
        smpl_data = np.load(smpl_file)
        return {
            "global_orient": smpl_data["global_orient"],
            "body_pose": smpl_data["body_pose"],
            "betas": smpl_data["betas"],
            "transl": smpl_data["transl"],
        }

    def _load_keypoints_3d(self, seq_name):
        keypoints_file = osp.join(self.data_root, "skl", f"{seq_name}_keypoints_3d.npz")
        keypoints_data = np.load(keypoints_file)
        return keypoints_data["keypoints_3d"]

    @staticmethod
    def _flatten_pose(global_orient, body_pose):
        global_orient = np.asarray(global_orient, dtype=np.float32).reshape(-1)
        body_pose = np.asarray(body_pose, dtype=np.float32).reshape(-1)
        return np.concatenate([global_orient, body_pose], axis=0)

    @staticmethod
    def _extract_pelvis(gt_keypoints, gt_transl):
        if gt_keypoints is not None:
            return np.asarray(gt_keypoints[0], dtype=np.float32)
        if gt_transl is not None:
            return np.asarray(gt_transl, dtype=np.float32).reshape(3)
        return np.zeros(3, dtype=np.float32)

    def _to_new_world(self, global_orient, pelvis, points):
        R_root = axis_angle_to_matrix_np(global_orient)
        return (R_root.T @ (points - pelvis).T).T

    def _update_extrinsic(self, R_wc, T_wc, R_root, pelvis):
        R_new = R_wc @ R_root
        T_new = R_wc @ pelvis.reshape(3, 1) + T_wc
        return R_new, T_new

    @staticmethod
    def _project_points(K: np.ndarray, R: np.ndarray, T: np.ndarray, points: np.ndarray) -> np.ndarray:
        cam = (R @ points.T + T).T
        z = np.clip(cam[:, 2:3], a_min=1e-6, a_max=None)
        pix = (K @ cam.T).T
        return pix[:, :2] / z

    @staticmethod
    def _bbox_from_keypoints(kp_2d: np.ndarray) -> np.ndarray:
        x_min = np.min(kp_2d[:, 0])
        y_min = np.min(kp_2d[:, 1])
        x_max = np.max(kp_2d[:, 0])
        y_max = np.max(kp_2d[:, 1])
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        size = max(w, h)
        c_x = (x_min + x_max) * 0.5
        c_y = (y_min + y_max) * 0.5
        return np.array([c_x, c_y, size, size], dtype=np.float32)

    def __call__(self, results):
        sample_id = results.get("sample_id", "")
        match = self._seq_re.match(sample_id)
        if match is None:
            raise ValueError(f"Unexpected sample_id format: {sample_id}")

        seq_name = match.group(1)
        rgb_camera = match.group(2)
        start_frame = int(match.group(4))

        if "input_rgb" in results:
            seq_len = len(results["input_rgb"])
        elif "input_depth" in results:
            seq_len = len(results["input_depth"])
        elif "input_lidar" in results:
            seq_len = len(results["input_lidar"])
        else:
            seq_len = 1

        smpl_params = self._load_smpl_params(seq_name)
        keypoints_3d = self._load_keypoints_3d(seq_name)
        cameras = self._load_camera_params(seq_name)

        if rgb_camera.startswith("kinect"):
            cam_key = f"kinect_color_{rgb_camera.split('_')[1]}"
        else:
            cam_key = "iphone"
        cam_params = cameras[cam_key]
        K = np.array(cam_params["K"], dtype=np.float32)
        R = np.array(cam_params["R"], dtype=np.float32)
        T = np.array(cam_params["T"], dtype=np.float32).reshape(3, 1)

        pose_seq = []
        shape_seq = []
        trans_seq = []
        joint_3d_pc_seq = []
        joint_3d_cam_seq = []
        joint_2d_seq = []

        for i in range(seq_len):
            idx = start_frame + i
            idx = min(idx, smpl_params["global_orient"].shape[0] - 1)
            global_orient = smpl_params["global_orient"][idx]
            body_pose = smpl_params["body_pose"][idx]
            betas = np.asarray(smpl_params["betas"][idx], dtype=np.float32)[:10]
            transl = np.asarray(smpl_params["transl"][idx], dtype=np.float32).reshape(3)

            keypoints_i = keypoints_3d[idx]
            pelvis = self._extract_pelvis(keypoints_i, transl)
            keypoints_new = self._to_new_world(global_orient, pelvis, keypoints_i)

            pose = self._flatten_pose(global_orient, body_pose)[:72]
            R_root = axis_angle_to_matrix_np(np.asarray(global_orient, dtype=np.float32))
            R_new, T_new = self._update_extrinsic(R, T, R_root, pelvis)

            joint_3d_pc = keypoints_new
            joint_3d_cam = (R_new @ keypoints_new.T + T_new).T
            joint_2d = self._project_points(K, R_new, T_new, keypoints_new)

            bbox = self._bbox_from_keypoints(joint_2d)
            joint_2d = transfrom_keypoints(
                joint_2d.copy(),
                center_x=bbox[0],
                center_y=bbox[1],
                width=bbox[2],
                height=bbox[3],
                patch_width=224,
                patch_height=224,
                scale=1.2,
            )
            joint_2d = normalize_2d_kp(joint_2d, crop_size=224)

            pose_seq.append(pose.astype(np.float32))
            shape_seq.append(betas.astype(np.float32))
            trans_seq.append(transl.astype(np.float32))
            joint_3d_pc_seq.append(joint_3d_pc.astype(np.float32))
            joint_3d_cam_seq.append(joint_3d_cam.astype(np.float32))
            joint_2d_seq.append(joint_2d.astype(np.float32))

        results["pose"] = np.stack(pose_seq, axis=0)
        results["shape"] = np.stack(shape_seq, axis=0)
        results["trans"] = np.stack(trans_seq, axis=0)
        results["joint_3d_pc"] = np.stack(joint_3d_pc_seq, axis=0)
        results["joint_3d_cam"] = np.stack(joint_3d_cam_seq, axis=0)
        results["joint_2d"] = np.stack(joint_2d_seq, axis=0)
        return results


class LEIRToTensor:
    def __init__(self):
        pass

    def _array_to_tensor(self, data, dtype=torch.float):
        return torch.from_numpy(data).to(dtype)

    def _item_to_tensor(self, data, dtype=torch.float):
        return torch.tensor([data], dtype=dtype)

    def _list_to_tensor(self, data, dtype=torch.float):
        return torch.from_numpy(np.stack(data, axis=0)).to(dtype)

    def __call__(self, sample):
        for key in sample:
            if key.endswith('_affine') or key in [
                'gt_keypoints',
                'gt_smpl_params',
                'pose',
                'shape',
                'trans',
                'joint_3d_pc',
                'joint_3d_cam',
                'joint_2d',
            ]:
                sample[key] = self._array_to_tensor(sample[key])
            elif key.startswith('input_'):
                sample[key] = self._list_to_tensor(sample[key])
                if key.startswith('input_rgb'):
                    sample[key] = sample[key].permute(0, 3, 1, 2)
                elif key.startswith('input_depth'):
                    if sample[key].ndim == 3:
                        sample[key] = sample[key].unsqueeze(1).repeat(1, 3, 1, 1)
                    else:
                        sample[key] = sample[key].permute(0, 3, 1, 2)
            elif key in ['idx']:
                sample[key] = self._item_to_tensor(sample[key], dtype=torch.long)
        return sample
