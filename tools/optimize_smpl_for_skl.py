import torch
import numpy as np
from rich.progress import track
import torch
import torch.nn.functional as F

import argparse
from copy import deepcopy

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from misc.utils import load, dump
import smplx

import os
import numpy as np
import torch
import torch.nn as nn
import smplx
import trimesh

class SMPLFitterFromKeypoints(nn.Module):
    """
    用 17 个 Human3.6M 格式的 3D 关键点拟合 SMPL mesh。
    - 关键点顺序假设为：
        0 pelvis
        1 r_hip,   2 r_knee,   3 r_ankle
        4 l_hip,   5 l_knee,   6 l_ankle
        7 spine,   8 thorax,   9 neck,  10 head
        11 l_shou, 12 l_elbow, 13 l_wrist
        14 r_shou, 15 r_elbow, 16 r_wrist
    """
    def __init__(
        self,
        model_folder: str,
        gender: str = "neutral",
        device: str = "cuda",
        num_betas: int = 10,
        batch_size: int = 1,
    ):
        super().__init__()
        self.device = torch.device(device)

        # 创建 SMPL 模型
        self.smpl = smplx.create(
            model_folder,
            model_type="smpl",
            gender=gender,
            use_face_contour=False,
            num_betas=num_betas,
            batch_size=batch_size,
        ).to(self.device)

        # Human3.6M 17 joints -> SMPL 24 joints 的索引映射
        # 如果你有自己确定好的映射，可以改这里
        # 0: pelvis
        # 1: left_hip, 4: left_knee, 7: left_ankle
        # 2: right_hip, 5: right_knee, 8: right_ankle
        # 3: spine1, 6: spine2, 12: neck, 15: head
        # 16: left_shoulder, 18: left_elbow, 20: left_wrist
        # 17: right_shoulder, 19: right_elbow, 21: right_wrist
        self.h36m_to_smpl_idx = torch.tensor(
            [
                0,          # pelvis
                2, 5, 8,    # right leg
                1, 4, 7,    # left leg
                3, 6, 12, 15,  # spine, thorax, neck, head
                13, 18, 20,    # left arm
                14, 19, 21,    # right arm
            ],
            dtype=torch.long,
            device=self.device,
        )

        # limb-length prior 用的骨骼边
        self.bone_pairs = [
            (0, 1), (1, 2), (2, 3),          # 右腿
            (0, 4), (4, 5), (5, 6),          # 左腿
            (0, 7), (7, 8), (8, 9), (9, 10), # 躯干到头
            (8, 11), (11, 12), (12, 13),     # 左臂
            (8, 14), (14, 15), (15, 16),     # 右臂
        ]

    def forward(self, *args, **kwargs):
        # 我们不在 forward 里做拟合，而是用专门的 fit() 函数
        raise NotImplementedError

    def fit(
        self,
        keypoints_3d: np.ndarray,
        num_iters: int = 800,
        lr: float = 0.01,
        joint_weight: float = 1.0,
        pose_reg: float = 1e-3,
        shape_reg: float = 1e-3,
        bone_len_reg: float = 1e-2,
        verbose: bool = True,
    ):
        """
        keypoints_3d: (17, 3) numpy，单位和 SMPL 无需完全一致，只要比例差不多即可
        """
        assert keypoints_3d.shape == (17, 3), f"expect (17,3), got {keypoints_3d.shape}"

        # -> torch: (1,17,3)
        keypoints = torch.from_numpy(keypoints_3d).float().to(self.device)[None, :, :]

        # 以 pelvis (0) 为中心，去掉全局平移
        keypoints_centered = keypoints - keypoints[:, 0:1, :]

        # ===== 可优化参数（注意 transl 形状是 (1,3)，不会再触发你遇到的 broadcast 报错）=====
        body_pose = torch.zeros(1, 23 * 3, device=self.device, requires_grad=True)
        global_orient = torch.zeros(1, 3, device=self.device, requires_grad=True)
        betas = torch.zeros(1, 10, device=self.device, requires_grad=True)
        transl = torch.zeros(1, 3, device=self.device, requires_grad=True)

        optim = torch.optim.Adam([body_pose, global_orient, betas, transl], lr=lr)

        for it in range(num_iters):
            optim.zero_grad()

            smpl_out = self.smpl(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )
            joints_smpl = smpl_out.joints  # (1, 45, 3) 之类
            joints_smpl = joints_smpl[:, :24, :]  # 只要前 24 个 SMPL 关节

            joints_17 = joints_smpl[:, self.h36m_to_smpl_idx, :]  # (1,17,3)
            joints_17_centered = joints_17 - joints_17[:, 0:1, :]

            # ---------- 关键点 L2 loss ----------
            joint_loss = ((joints_17_centered - keypoints_centered) ** 2).sum(-1).mean()

            # ---------- 姿态 / 形状 正则 ----------
            pose_loss = (body_pose ** 2).mean()
            shape_loss = (betas ** 2).mean()

            # ---------- limb-length prior ----------
            with torch.no_grad():
                gt_bone_len = []
                for i, j in self.bone_pairs:
                    gt_bone_len.append(
                        (keypoints_centered[:, i] - keypoints_centered[:, j]).norm(dim=-1)
                    )
                gt_bone_len = torch.stack(gt_bone_len, dim=-1)  # (1, num_bones)
                gt_bone_len = gt_bone_len / (gt_bone_len.mean(dim=-1, keepdim=True) + 1e-8)

            pred_bone_len = []
            for i, j in self.bone_pairs:
                pred_bone_len.append(
                    (joints_17_centered[:, i] - joints_17_centered[:, j]).norm(dim=-1)
                )
            pred_bone_len = torch.stack(pred_bone_len, dim=-1)
            pred_bone_len = pred_bone_len / (pred_bone_len.mean(dim=-1, keepdim=True) + 1e-8)

            bone_loss = ((pred_bone_len - gt_bone_len) ** 2).mean()

            total_loss = (
                joint_weight * joint_loss
                + pose_reg * pose_loss
                + shape_reg * shape_loss
                + bone_len_reg * bone_loss
            )

            total_loss.backward()
            optim.step()

            if verbose and (it % 50 == 0 or it == num_iters - 1):
                print(
                    f"[Iter {it:04d}] "
                    f"total_loss={total_loss.item():.4f}, "
                    f"joint_loss={joint_loss.item():.4f}, "
                    f"pose={pose_loss.item():.4f}, "
                    f"shape={shape_loss.item():.4f}, "
                    f"bone={bone_loss.item():.4f}"
                )

        with torch.no_grad():
            smpl_out = self.smpl(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )
            verts = smpl_out.vertices[0].cpu()
            joints = smpl_out.joints[0].cpu()

        return verts, joints

    def fit_sequence(
        self,
        keypoints_seq_3d: np.ndarray,   # (T,17,3)
        num_iters: int = 1000,
        lr: float = 0.02,
        joint_weight: float = 1.0,
        pose_reg: float = 0.02,
        shape_reg: float = 0.1,
        bone_len_reg: float = 0.01,
        verbose: bool = True,
    ):
        """
        批量拟合一段序列：
        keypoints_seq_3d: (T,17,3) 的 numpy，T 是帧数
        - 所有帧共享一个 betas（身体形状），
        - 每一帧有自己的 body_pose / global_orient / transl。
        """
        assert (
            keypoints_seq_3d.ndim == 3
            and keypoints_seq_3d.shape[1:] == (17, 3)
        ), f"expect (T,17,3), got {keypoints_seq_3d.shape}"

        T = keypoints_seq_3d.shape[0]

        # (T,17,3) -> torch
        keypoints = torch.from_numpy(keypoints_seq_3d).float().to(self.device)
        keypoints[:, 0, :] -= torch.tensor([0., 0.1, 0.]).to(self.device)  # pelvis 下移 0.1m，更符合 SMPL 初始姿态

        # 每一帧都减去 pelvis，去全局平移
        keypoints_centered = keypoints - keypoints[:, 0:1, :]

        # ===== 可优化参数 =====
        # 每一帧一套 pose / orient / transl
        body_pose = torch.zeros(T, 23 * 3, device=self.device, requires_grad=True)
        global_orient = torch.zeros(T, 3, device=self.device, requires_grad=True)
        transl = torch.zeros(T, 3, device=self.device, requires_grad=True)

        # 所有帧共享一个 betas（身体形状）
        betas = torch.zeros(1, 10, device=self.device, requires_grad=True)

        optim = torch.optim.Adam(
            [body_pose, global_orient, transl, betas],
            lr=lr
        )

        for it in range(num_iters):
            optim.zero_grad()

            # 将 betas 扩展到 T 帧
            betas_expanded = betas.expand(T, -1)

            smpl_out = self.smpl(
                betas=betas_expanded,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )
            # joints_smpl: (T, J, 3)
            joints_smpl = smpl_out.joints[:, :24, :]  # 只取前 24 个关节
            joints_17 = joints_smpl[:, self.h36m_to_smpl_idx, :]  # (T,17,3)
            joints_17_centered = joints_17 - joints_17[:, 0:1, :]

            # ---------- 关键点 L2 loss ----------
            joint_loss = ((joints_17_centered - keypoints_centered) ** 2).sum(-1).mean()

            # ---------- 姿态 / 形状 正则 ----------
            pose_loss = (body_pose ** 2).mean()
            shape_loss = (betas ** 2).mean()

            # ---------- limb-length prior ----------
            with torch.no_grad():
                gt_bone_len = []
                for i, j in self.bone_pairs:
                    gt_bone_len.append(
                        (keypoints_centered[:, i] - keypoints_centered[:, j]).norm(dim=-1)
                    )
                # (T, num_bones)
                gt_bone_len = torch.stack(gt_bone_len, dim=-1)
                gt_bone_len = gt_bone_len / (
                    gt_bone_len.mean(dim=-1, keepdim=True) + 1e-8
                )

            pred_bone_len = []
            for i, j in self.bone_pairs:
                pred_bone_len.append(
                    (joints_17_centered[:, i] - joints_17_centered[:, j]).norm(dim=-1)
                )
            pred_bone_len = torch.stack(pred_bone_len, dim=-1)
            pred_bone_len = pred_bone_len / (
                pred_bone_len.mean(dim=-1, keepdim=True) + 1e-8
            )

            bone_loss = ((pred_bone_len - gt_bone_len) ** 2).mean()

            total_loss = (
                joint_weight * joint_loss
                + pose_reg * pose_loss
                + shape_reg * shape_loss
                + bone_len_reg * bone_loss
            )
            # total_loss = joint_weight * joint_loss

            total_loss.backward()
            optim.step()

            if verbose and (it % 50 == 0 or it == num_iters - 1):
                print(
                    f"[Seq Iter {it:04d}] "
                    f"total={total_loss.item():.4f}, "
                    f"joint={joint_loss.item():.4f}, "
                    f"pose={pose_loss.item():.4f}, "
                    f"shape={shape_loss.item():.4f}, "
                    f"bone={bone_loss.item():.4f}"
                )

        # ===== 返回所有帧的 verts / joints =====
        with torch.no_grad():
            betas_expanded = betas.expand(T, -1)
            smpl_out = self.smpl(
                betas=betas_expanded,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )
            verts = smpl_out.vertices.cpu()  # (T, V, 3)
            joints = smpl_out.joints.cpu()   # (T, J, 3)

            params = {
                'body_pose': body_pose.detach().cpu().numpy(),
                'global_orient': global_orient.detach().cpu().numpy(),
                'betas': betas.detach().cpu().numpy(),
                'transl': transl.detach().cpu().numpy(),
            }

        return verts, joints, params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', type=str, required=True, help='Path to predictions pkl file')
    parser.add_argument('--model_folder', type=str, default='/home/zpengac/mmhpe/MmMvHPE/weights', help='Path to SMPL model folder')
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for fitting')
    parser.add_argument('--num_iters', type=int, default=2000, help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--one_batch', action='store_true', help='Only process one batch for debugging')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preds = load(args.preds_path)
    outputs = deepcopy(preds)
    
    # Initialize SMPL fitter
    fitter = SMPLFitterFromKeypoints(
        model_folder=args.model_folder,
        gender=args.gender,
        device=device,
        num_betas=10,
        batch_size=args.batch_size,
    )

    for prefix in ['pred_', 'gt_']:
        key = prefix + 'keypoints'
        if preds.get(key) is not None:
            print(f"\nFitting {key}...")
            keypoints_data = preds[key]  # (N, 17, 3)
            num_samples = keypoints_data.shape[0]

            all_poses = []
            all_global_orients = []
            all_betas = []
            all_transls = []
            
            for i in range(0, num_samples, args.batch_size):
                batch_end = min(i + args.batch_size, num_samples)
                print(f"\nProcessing batch {i//args.batch_size + 1}/{(num_samples-1)//args.batch_size + 1} (samples {i}-{batch_end-1})")
                
                kps_batch = keypoints_data[i:batch_end]  # (batch_size, 17, 3)
                
                # Fit the sequence
                verts, joints, params = fitter.fit_sequence(
                    keypoints_seq_3d=kps_batch,
                    num_iters=args.num_iters,
                    lr=args.lr,
                    verbose=True,
                )
                
                all_poses.append(params['body_pose'])
                all_global_orients.append(params['global_orient'])
                all_betas.append(np.repeat(params['betas'], repeats=verts.shape[0], axis=0))
                all_transls.append(params['transl'])
                
                if args.one_batch:
                    break
                    
            outputs[prefix + 'pose'] = np.concatenate(all_poses, axis=0)
            outputs[prefix + 'global_orient'] = np.concatenate(all_global_orients, axis=0)
            outputs[prefix + 'beta'] = np.concatenate(all_betas, axis=0)
            outputs[prefix + 'translation'] = np.concatenate(all_transls, axis=0)

    dump(outputs, args.preds_path.replace('.pkl', '_smpl.pkl'))
    print("\nFitting finished! Saved to:", args.preds_path.replace('.pkl', '_smpl.pkl'))