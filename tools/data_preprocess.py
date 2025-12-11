from glob import glob
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _process_single_sample(args):
    """Worker function to process one RGB/Depth/LiDAR/mmWave sample."""
    rgb_fn, depth_fn, lidar_fn, mmwave_fn, out_dir, rgb_out_size, depth_out_size = args
    # gt_fn: parent of parent of rgb_fn + 'ground_truth.npy'

    E, S, A, _, basename = rgb_fn.split(os.sep)[-5:]
    basename = basename.split('.')[0]
    new_name = f'{E}_{S}_{A}_{basename}'

    out_rgb_fn = osp.join(out_dir, 'rgb', new_name + '.jpg')
    out_depth_fn = out_rgb_fn.replace('rgb', 'depth')
    out_lidar_fn = out_rgb_fn.replace('rgb', 'lidar').replace('.jpg', '.npy')
    out_mmwave_fn = out_rgb_fn.replace('rgb', 'mmwave').replace('.jpg', '.npy')

    # Skip if all outputs already exist (idempotent)
    if os.path.exists(out_rgb_fn) and os.path.exists(out_depth_fn) \
            and os.path.exists(out_lidar_fn) and os.path.exists(out_mmwave_fn):
        return

    # Process RGB
    rgb_img = cv2.imread(rgb_fn, cv2.IMREAD_COLOR)
    if rgb_img is None:
        return
    rgb_img = cv2.resize(rgb_img, rgb_out_size)
    cv2.imwrite(out_rgb_fn, rgb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Process Depth
    depth_img = cv2.imread(depth_fn)
    if depth_img is None:
        return
    depth_img = cv2.resize(depth_img, depth_out_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_depth_fn, depth_img)

    # Process LiDAR
    lidar_points = np.fromfile(lidar_fn, dtype=np.float64).reshape(-1, 3)
    np.save(out_lidar_fn, lidar_points.astype(np.float16))

    # Process mmWave
    mmwave_points = np.fromfile(mmwave_fn, dtype=np.float64).reshape(-1, 5)
    np.save(out_mmwave_fn, mmwave_points.astype(np.float16))


def preprocess_mmfi(root_dir, rgb_dir, out_dir, rgb_out_size, depth_out_size, num_workers=None):
    rgb_fns = glob(osp.join(rgb_dir, 'E*/S*/A*/rgb/*.png'))
    depth_fns = [fn.replace(rgb_dir, root_dir).replace('rgb', 'depth') for fn in rgb_fns]
    lidar_fns = [fn.replace('depth', 'lidar').replace('.png', '.bin') for fn in depth_fns]
    mmwave_fns = [fn.replace('lidar', 'mmwave') for fn in lidar_fns]

    os.makedirs(out_dir, exist_ok=True)

    modalities = ['rgb', 'depth', 'lidar', 'mmwave']
    for modality in modalities:
        out_modality_dir = osp.join(out_dir, modality)
        os.makedirs(out_modality_dir, exist_ok=True)

    args_list = [
        (rgb_fn, depth_fn, lidar_fn, mmwave_fn, out_dir, rgb_out_size, depth_out_size)
        for rgb_fn, depth_fn, lidar_fn, mmwave_fn in zip(rgb_fns, depth_fns, lidar_fns, mmwave_fns)
    ]

    if num_workers is None or num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_process_single_sample, args_list), total=len(args_list)):
            pass

    gt_fns = glob(osp.join(root_dir, 'E*/S*/A*/ground_truth.npy'))
    out_gt_dir = osp.join(out_dir, 'gt')
    os.makedirs(out_gt_dir, exist_ok=True)
    for gt_fn in gt_fns:
        E, S, A, _ = gt_fn.split(os.sep)[-4:]
        new_name = f'{E}_{S}_{A}.npy'
        out_gt_fn = osp.join(out_gt_dir, new_name)
        if os.path.exists(out_gt_fn):
            continue
        gt_data = np.load(gt_fn)
        np.save(out_gt_fn, gt_data.astype(np.float16))


if __name__ == '__main__':
    root_dir = '/data/shared/MMFi_Dataset'
    rgb_dir = '/data/shared/MMFi_Defaced_RGB'
    out_dir = 'data/mmfi'
    rgb_out_size = (256, 192)
    depth_out_size = (256, 192)

    preprocess_mmfi(root_dir, rgb_dir, out_dir, rgb_out_size, depth_out_size, num_workers=16)
