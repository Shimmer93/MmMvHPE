from glob import glob
import os
import os.path as osp
import cv2
import numpy as np
import json
import shutil
import tempfile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _process_single_sample_mmfi(args):
    """Worker function to process one RGB/Depth/LiDAR/mmWave sample."""
    rgb_fn, depth_fn, lidar_fn, mmwave_fn, out_dir, rgb_out_size, depth_out_size = args
    # gt_fn: parent of parent of rgb_fn + 'ground_truth.npy'

    E, S, A, _, basename = rgb_fn.split(os.sep)[-5:]
    basename = basename.split('.')[0]
    new_name = f'{E}_{S}_{A}_{basename}'

    out_rgb_fn = osp.join(out_dir, 'rgb', new_name + '.jpg')
    out_depth_fn = out_rgb_fn.replace('rgb', 'depth').replace('.jpg', '.png')
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
        for _ in tqdm(pool.imap_unordered(_process_single_sample_mmfi, args_list), total=len(args_list)):
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

def _process_single_sample_humman(args):
    """Worker function to process one RGB/Depth sample."""
    rgb_fn, depth_fn, out_dir, rgb_out_size, depth_out_size = args

    p1, p2, p3, p4, basename = rgb_fn.split(os.sep)[-5:]
    basename = basename.split('.')[0]
    new_name = f'{p2}_{p4}_{basename}'
    out_rgb_fn = osp.join(out_dir, 'rgb', new_name + '.jpg')
    out_depth_fn = out_rgb_fn.replace('rgb', 'depth').replace('.jpg', '.png')

    # Skip if all outputs already exist (idempotent)
    if os.path.exists(out_rgb_fn) and os.path.exists(out_depth_fn):
        return
    
    # Process RGB (use reduced decode when heavily downscaling)
    max_out = max(rgb_out_size)
    if max_out <= 512:
        rgb_read_flag = cv2.IMREAD_REDUCED_COLOR_4
    elif max_out <= 1024:
        rgb_read_flag = cv2.IMREAD_REDUCED_COLOR_2
    else:
        rgb_read_flag = cv2.IMREAD_COLOR
    rgb_img = cv2.imread(rgb_fn, rgb_read_flag)
    if rgb_img is None:
        print( f"Warning: failed to read {rgb_fn}" )
        return
    rgb_img = cv2.resize(rgb_img, rgb_out_size)
    cv2.imwrite(out_rgb_fn, rgb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Process Depth
    depth_img = cv2.imread(depth_fn)
    if depth_img is None:
        print( f"Warning: failed to read {depth_fn}" )
        return
    depth_img = cv2.resize(depth_img, depth_out_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_depth_fn, depth_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def _find_first_image_size(dir_path):
    img_fns = glob(osp.join(dir_path, '*.png'))
    if not img_fns:
        return None
    img = cv2.imread(img_fns[0], cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    return (w, h)

KINECT_COLOR_SIZE = (1920, 1080)
KINECT_DEPTH_SIZE = (640, 576)


def _scale_camera_intrinsic(K, scale_x, scale_y):
    K = K.copy()
    K[0, :] *= scale_x
    K[1, :] *= scale_y
    return K


def _resize_camera_params(camera_params, seq_dir, rgb_out_size, depth_out_size):
    resized = {}
    size_cache = {}
    for cam_key, cam in camera_params.items():
        K = np.array(cam.get('K', []), dtype=np.float32)
        if K.size == 0:
            resized[cam_key] = cam
            continue

        if cam_key.startswith('kinect_color_'):
            src_size = KINECT_COLOR_SIZE
            dst_size = rgb_out_size
        elif cam_key.startswith('kinect_depth_'):
            src_size = KINECT_DEPTH_SIZE
            dst_size = depth_out_size
        elif cam_key == 'iphone':
            src_dir = osp.join(seq_dir, 'iphone_color', 'iphone')
            dst_size = rgb_out_size
            if not osp.exists(src_dir):
                src_dir = osp.join(seq_dir, 'iphone_depth', 'iphone')
                dst_size = depth_out_size
        else:
            resized[cam_key] = cam
            continue

        if cam_key == 'iphone':
            if src_dir in size_cache:
                src_size = size_cache[src_dir]
            else:
                src_size = _find_first_image_size(src_dir)
                size_cache[src_dir] = src_size

        if src_size is None:
            resized[cam_key] = cam
            continue

        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        scale_x = float(dst_w) / float(src_w)
        scale_y = float(dst_h) / float(src_h)
        K_scaled = _scale_camera_intrinsic(K, scale_x, scale_y)

        cam_scaled = dict(cam)
        cam_scaled['K'] = K_scaled.tolist()
        resized[cam_key] = cam_scaled

    return resized


def _iter_chunks(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _stage_sequence_dirs(seq_dirs, stage_root):
    subdirs_to_copy = ['kinect_color', 'kinect_depth']
    for seq_dir in seq_dirs:
        seq_name = osp.basename(seq_dir)
        dst_seq_dir = osp.join(stage_root, seq_name)
        os.makedirs(dst_seq_dir, exist_ok=True)
        for subdir in subdirs_to_copy:
            src_subdir = osp.join(seq_dir, subdir)
            if not osp.exists(src_subdir):
                continue
            dst_subdir = osp.join(dst_seq_dir, subdir)
            shutil.copytree(
                src_subdir,
                dst_subdir,
                dirs_exist_ok=True,
                copy_function=shutil.copyfile,
            )


def _preprocess_humman_root(root_dir, out_dir, rgb_out_size, depth_out_size, num_workers):
    rgb_fns = glob(osp.join(root_dir, 'p*/kinect_color/kinect_*/*.png'))
    depth_fns = [fn.replace('kinect_color', 'kinect_depth') for fn in rgb_fns]

    os.makedirs(out_dir, exist_ok=True)

    modalities = ['rgb', 'depth']
    for modality in modalities:
        out_modality_dir = osp.join(out_dir, modality)
        os.makedirs(out_modality_dir, exist_ok=True)

    args_list = [
        (rgb_fn, depth_fn, out_dir, rgb_out_size, depth_out_size)
        for rgb_fn, depth_fn in zip(rgb_fns, depth_fns)
    ]

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_process_single_sample_humman, args_list), total=len(args_list)):
            pass

    camera_fns = glob(osp.join(root_dir, 'p*/cameras.json'))
    out_camera_dir = osp.join(out_dir, 'cameras')
    os.makedirs(out_camera_dir, exist_ok=True)
    for camera_fn in tqdm(camera_fns):
        p = camera_fn.split(os.sep)[-2]
        new_name = f'{p}_cameras.json'
        out_camera_fn = osp.join(out_camera_dir, new_name)
        if os.path.exists(out_camera_fn):
            continue
        with open(camera_fn, 'r') as f:
            cameras = json.load(f)
        cameras = _resize_camera_params(cameras, osp.dirname(camera_fn), rgb_out_size, depth_out_size)
        with open(out_camera_fn, 'w') as f:
            json.dump(cameras, f)

    skl_fns = glob(osp.join(root_dir, 'p*/keypoints_3d.npz'))
    out_skl_dir = osp.join(out_dir, 'skl')
    os.makedirs(out_skl_dir, exist_ok=True)
    for skl_fn in tqdm(skl_fns):
        p = skl_fn.split(os.sep)[-2]
        new_name = f'{p}_keypoints_3d.npz'
        out_skl_fn = osp.join(out_skl_dir, new_name)
        if os.path.exists(out_skl_fn):
            continue
        shutil.copyfile(skl_fn, out_skl_fn)

    smpl_fns = glob(osp.join(root_dir, 'p*/smpl_params.npz'))
    out_smpl_dir = osp.join(out_dir, 'smpl')
    os.makedirs(out_smpl_dir, exist_ok=True)
    for smpl_fn in tqdm(smpl_fns):
        p = smpl_fn.split(os.sep)[-2]
        new_name = f'{p}_smpl_params.npz'
        out_smpl_fn = osp.join(out_smpl_dir, new_name)
        if os.path.exists(out_smpl_fn):
            continue
        shutil.copyfile(smpl_fn, out_smpl_fn)


def preprocess_humman(
    root_dir,
    out_dir,
    rgb_out_size,
    depth_out_size,
    num_workers=None,
    staging_dir=None,
    staging_chunk_size=5,
):
    if num_workers is None or num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    if staging_dir is None:
        _preprocess_humman_root(root_dir, out_dir, rgb_out_size, depth_out_size, num_workers)
    else:
        seq_dirs = sorted(glob(osp.join(root_dir, 'p*_a*')))
        chunk_iter = _iter_chunks(seq_dirs, staging_chunk_size)
        for chunk_idx, chunk in enumerate(tqdm(chunk_iter, total=(len(seq_dirs) + staging_chunk_size - 1) // staging_chunk_size), start=1):
            stage_root = tempfile.mkdtemp(prefix=f"humman_stage_{chunk_idx}_", dir=staging_dir)
            _stage_sequence_dirs(chunk, stage_root)
            _preprocess_humman_root(stage_root, out_dir, rgb_out_size, depth_out_size, num_workers)
            shutil.rmtree(stage_root, ignore_errors=True)

if __name__ == '__main__':
    # root_dir = '/data/shared/MMFi_Dataset'
    # rgb_dir = '/data/shared/MMFi_Defaced_RGB'
    # out_dir = 'data/mmfi'
    # rgb_out_size = (256, 192)
    # depth_out_size = (256, 192)

    # preprocess_mmfi(root_dir, rgb_dir, out_dir, rgb_out_size, depth_out_size, num_workers=16)

    root_dir = '/data/shared/humman_release_v1.0_point'
    out_dir = '/opt/data/humman'
    rgb_out_size = (320, 180)
    depth_out_size = (320, 288)

    preprocess_humman(root_dir, out_dir, rgb_out_size, depth_out_size, num_workers=16, staging_dir=None)
    # preprocess_humman(root_dir, out_dir, rgb_out_size, depth_out_size, num_workers=16, staging_dir='data/temp')
