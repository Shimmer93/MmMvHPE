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

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

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


def _adjust_camera_intrinsic_for_crop(K, crop_x0, crop_y0, crop_size, out_size):
    out_w, out_h = out_size
    scale_x = float(out_w) / float(crop_size)
    scale_y = float(out_h) / float(crop_size)
    K_new = K.copy()
    K_new[0, 0] *= scale_x
    K_new[1, 1] *= scale_y
    K_new[0, 2] = (K_new[0, 2] - crop_x0) * scale_x
    K_new[1, 2] = (K_new[1, 2] - crop_y0) * scale_y
    K_new[0, 1] *= scale_x
    K_new[1, 0] *= scale_y
    return K_new


def _compute_square_crop(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    size = int(np.ceil(max(w, h)))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    x0 = int(np.floor(cx - size * 0.5))
    y0 = int(np.floor(cy - size * 0.5))
    x1 = x0 + size
    y1 = y0 + size
    return x0, y0, x1, y1, size


def _crop_with_padding(image, x0, y0, x1, y1, pad_value=0):
    h, w = image.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    x0_clamped = max(0, x0)
    y0_clamped = max(0, y0)
    x1_clamped = min(w, x1)
    y1_clamped = min(h, y1)
    cropped = image[y0_clamped:y1_clamped, x0_clamped:x1_clamped]
    if pad_left or pad_top or pad_right or pad_bottom:
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )
    return cropped


def _depth_to_lidar_pc(depth_m, K, extrinsic, min_depth=1e-6):
    H, W = depth_m.shape
    xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_m.reshape(-1)
    valid = z > min_depth
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    K_inv = np.linalg.inv(K)
    pixels = np.stack([xmap.reshape(-1), ymap.reshape(-1), np.ones(H * W)], axis=0)
    rays = K_inv @ pixels
    cam_points = rays * z
    cam_points = cam_points[:, valid]
    R = extrinsic[:, :3]
    T = extrinsic[:, 3:].reshape(3, 1)
    world_points = (R.T @ (cam_points - T)).T
    return world_points.astype(np.float32)


def _iter_chunks(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _stage_sequence_dirs(seq_dirs, stage_root):
    subdirs_to_copy = ['kinect_color', 'kinect_depth', 'kinect_mask']
    # subdirs_to_copy = ['kinect_color', 'kinect_depth', 'kinect_mask', 'iphone_color', 'iphone_depth']
    for seq_dir in seq_dirs:
        seq_name = osp.basename(seq_dir)
        dst_seq_dir = osp.join(stage_root, seq_name)
        os.makedirs(dst_seq_dir, exist_ok=True)
        for filename in ['cameras.json', 'keypoints_3d.npz', 'smpl_params.npz']:
            src_file = osp.join(seq_dir, filename)
            if osp.exists(src_file):
                shutil.copyfile(src_file, osp.join(dst_seq_dir, filename))
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
    # kinect_rgb_fns = glob(osp.join(root_dir, 'p*/kinect_color/kinect_*/*.png'))
    # kinect_depth_fns = [fn.replace('kinect_color', 'kinect_depth') for fn in kinect_rgb_fns]

    iphone_rgb_fns = glob(osp.join(root_dir, 'p*/iphone_color/iphone/*.png'))
    iphone_depth_fns = [fn.replace('iphone_color', 'iphone_depth') for fn in iphone_rgb_fns]

    rgb_fns = iphone_rgb_fns
    depth_fns = iphone_depth_fns
    # rgb_fns = kinect_rgb_fns + iphone_rgb_fns
    # depth_fns = kinect_depth_fns + iphone_depth_fns

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

    camera_fns = glob(osp.join(root_dir, 'p*_a*/cameras.json'))
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


def _process_single_sample_humman_cropped(args):
    (
        rgb_fn,
        depth_fn,
        mask_fn,
        out_dir,
        rgb_out_size,
        depth_out_size,
        rgb_crop,
        depth_crop,
        depth_cam_params,
    ) = args

    p1, p2, p3, p4, basename = rgb_fn.split(os.sep)[-5:]
    basename = basename.split('.')[0]
    new_name = f'{p2}_{p4}_{basename}'
    out_rgb_fn = osp.join(out_dir, 'rgb', new_name + '.jpg')
    out_depth_fn = out_rgb_fn.replace('rgb', 'depth').replace('.jpg', '.png')
    out_lidar_fn = out_rgb_fn.replace('rgb', 'lidar').replace('.jpg', '.npy')

    if os.path.exists(out_rgb_fn) and os.path.exists(out_depth_fn) and os.path.exists(out_lidar_fn):
        return

    rgb_img = cv2.imread(rgb_fn, cv2.IMREAD_COLOR)
    if rgb_img is None:
        return
    rx0, ry0, rx1, ry1, _ = rgb_crop
    rgb_crop_img = _crop_with_padding(rgb_img, rx0, ry0, rx1, ry1, pad_value=(0, 0, 0))
    rgb_crop_img = cv2.resize(rgb_crop_img, rgb_out_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_rgb_fn, rgb_crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    depth_img = cv2.imread(depth_fn, cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        return
    mask_img = cv2.imread(mask_fn, cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        return
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]
    dx0, dy0, dx1, dy1, _ = depth_crop
    depth_crop_img = _crop_with_padding(depth_img, dx0, dy0, dx1, dy1, pad_value=0)
    mask_crop = _crop_with_padding(mask_img, dx0, dy0, dx1, dy1, pad_value=0)
    mask_crop = mask_crop > 0
    depth_crop_img = depth_crop_img.copy()
    depth_crop_img[~mask_crop] = 0
    depth_crop_img = cv2.resize(depth_crop_img, depth_out_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_depth_fn, depth_crop_img)

    depth_m = depth_crop_img.astype(np.float32) / 1000.0
    lidar_pc = _depth_to_lidar_pc(
        depth_m,
        depth_cam_params['K'],
        depth_cam_params['extrinsic'],
    )
    np.save(out_lidar_fn, lidar_pc.astype(np.float16))


def _compute_mask_bbox_union(mask_dir):
    mask_fns = sorted(glob(osp.join(mask_dir, '*.png')))
    if not mask_fns:
        return None
    min_x = min_y = None
    max_x = max_y = None
    for mask_fn in mask_fns:
        mask = cv2.imread(mask_fn, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        min_x = x0 if min_x is None else min(min_x, x0)
        min_y = y0 if min_y is None else min(min_y, y0)
        max_x = x1 if max_x is None else max(max_x, x1)
        max_y = y1 if max_y is None else max(max_y, y1)
    if min_x is None:
        return None
    return [float(min_x), float(min_y), float(max_x), float(max_y)]


def _preprocess_humman_cropped_root(
    root_dir,
    out_dir,
    rgb_out_size,
    depth_out_size,
    num_workers,
    yolo_conf=0.25,
    yolo_iou=0.45,
    yolo_device=None,
):
    from tools.image_detection import predict_human_bboxes
    from ultralytics import YOLO

    rgb_fns = glob(osp.join(root_dir, 'p*_a*/kinect_color/kinect_*/*.png'))
    rgb_fns = sorted(rgb_fns)
    if not rgb_fns:
        print(f"Warning: no kinect RGB frames found under {root_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    for modality in ['rgb', 'depth', 'lidar']:
        os.makedirs(osp.join(out_dir, modality), exist_ok=True)

    seq_dirs = sorted(set(osp.dirname(osp.dirname(osp.dirname(fn))) for fn in rgb_fns))
    yolo_model = YOLO('yolov8n.pt')

    crop_params = {}
    camera_updates = {}
    for seq_dir in tqdm(seq_dirs, desc='Compute crops'):
        seq_name = osp.basename(seq_dir)
        cam_dirs = sorted(glob(osp.join(seq_dir, 'kinect_color', 'kinect_*')))
        if not cam_dirs:
            continue
        cam_file = osp.join(seq_dir, 'cameras.json')
        if not osp.exists(cam_file):
            continue
        with open(cam_file, 'r') as f:
            cameras = json.load(f)
        updated_cameras = dict(cameras)

        for cam_dir in cam_dirs:
            cam_name = osp.basename(cam_dir)
            sample_rgb_fns = sorted(glob(osp.join(cam_dir, '*.png')))
            if not sample_rgb_fns:
                continue
            rgb_img = cv2.imread(sample_rgb_fns[0], cv2.IMREAD_COLOR)
            if rgb_img is None:
                continue
            img_h, img_w = rgb_img.shape[:2]
            bboxes = predict_human_bboxes(
                image_path=sample_rgb_fns[0],
                model=yolo_model,
                conf=yolo_conf,
                iou=yolo_iou,
                device=yolo_device,
            )
            if bboxes:
                rgb_bbox = max(bboxes, key=lambda b: b['score'])['bbox']
            else:
                rgb_bbox = [0.0, 0.0, float(img_w - 1), float(img_h - 1)]

            mask_dir = osp.join(seq_dir, 'kinect_mask', cam_name)
            depth_bbox = _compute_mask_bbox_union(mask_dir)
            if depth_bbox is None:
                depth_bbox = [0.0, 0.0, float(KINECT_DEPTH_SIZE[0] - 1), float(KINECT_DEPTH_SIZE[1] - 1)]

            rx0, ry0, rx1, ry1, rsize = _compute_square_crop(rgb_bbox, img_w, img_h)
            dx0, dy0, dx1, dy1, dsize = _compute_square_crop(
                depth_bbox,
                KINECT_DEPTH_SIZE[0],
                KINECT_DEPTH_SIZE[1],
            )

            crop_params[(seq_name, cam_name)] = {
                'rgb_crop': (rx0, ry0, rx1, ry1, rsize),
                'depth_crop': (dx0, dy0, dx1, dy1, dsize),
            }

            cam_suffix = cam_name.split('_')[1]
            rgb_key = f'kinect_color_{cam_suffix}'
            depth_key = f'kinect_depth_{cam_suffix}'
            if rgb_key in updated_cameras:
                K_rgb = np.array(updated_cameras[rgb_key]['K'], dtype=np.float32)
                K_rgb = _adjust_camera_intrinsic_for_crop(K_rgb, rx0, ry0, rsize, rgb_out_size)
                cam_out = dict(updated_cameras[rgb_key])
                cam_out['K'] = K_rgb.tolist()
                updated_cameras[rgb_key] = cam_out
            if depth_key in updated_cameras:
                K_depth = np.array(updated_cameras[depth_key]['K'], dtype=np.float32)
                K_depth = _adjust_camera_intrinsic_for_crop(K_depth, dx0, dy0, dsize, depth_out_size)
                cam_out = dict(updated_cameras[depth_key])
                cam_out['K'] = K_depth.tolist()
                updated_cameras[depth_key] = cam_out

        camera_updates[seq_name] = updated_cameras

    camera_fns = glob(osp.join(root_dir, 'p*/cameras.json'))
    out_camera_dir = osp.join(out_dir, 'cameras')
    os.makedirs(out_camera_dir, exist_ok=True)
    for camera_fn in tqdm(camera_fns, desc='Write cameras'):
        seq_name = camera_fn.split(os.sep)[-2]
        out_camera_fn = osp.join(out_camera_dir, f'{seq_name}_cameras.json')
        if os.path.exists(out_camera_fn):
            continue
        cameras = camera_updates.get(seq_name)
        if cameras is None:
            with open(camera_fn, 'r') as f:
                cameras = json.load(f)
        with open(out_camera_fn, 'w') as f:
            json.dump(cameras, f)

    rgb_fns = glob(osp.join(root_dir, 'p*_a*/kinect_color/kinect_*/*.png'))
    depth_fns = [fn.replace('kinect_color', 'kinect_depth') for fn in rgb_fns]
    mask_fns = [fn.replace('kinect_color', 'kinect_mask') for fn in rgb_fns]

    args_list = []
    for rgb_fn, depth_fn, mask_fn in zip(rgb_fns, depth_fns, mask_fns):
        if not osp.exists(depth_fn) or not osp.exists(mask_fn):
            continue
        seq_dir = osp.dirname(osp.dirname(osp.dirname(rgb_fn)))
        seq_name = osp.basename(seq_dir)
        cam_name = osp.basename(osp.dirname(rgb_fn))
        crop = crop_params.get((seq_name, cam_name))
        if crop is None:
            continue
        cam_suffix = cam_name.split('_')[1]
        depth_key = f'kinect_depth_{cam_suffix}'
        cam_info = camera_updates.get(seq_name, {})
        if depth_key not in cam_info:
            continue
        depth_cam = cam_info[depth_key]
        K_depth = np.array(depth_cam['K'], dtype=np.float32)
        extrinsic = np.hstack(
            [
                np.array(depth_cam['R'], dtype=np.float32),
                np.array(depth_cam['T'], dtype=np.float32).reshape(3, 1),
            ]
        ).astype(np.float32)
        args_list.append(
            (
                rgb_fn,
                depth_fn,
                mask_fn,
                out_dir,
                rgb_out_size,
                depth_out_size,
                crop['rgb_crop'],
                crop['depth_crop'],
                {'K': K_depth, 'extrinsic': extrinsic},
            )
        )

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_process_single_sample_humman_cropped, args_list), total=len(args_list)):
            pass

    skl_fns = glob(osp.join(root_dir, 'p*_a*/keypoints_3d.npz'))
    out_skl_dir = osp.join(out_dir, 'skl')
    os.makedirs(out_skl_dir, exist_ok=True)
    for skl_fn in tqdm(skl_fns, desc='Copy skl'):
        p = skl_fn.split(os.sep)[-2]
        new_name = f'{p}_keypoints_3d.npz'
        out_skl_fn = osp.join(out_skl_dir, new_name)
        if os.path.exists(out_skl_fn):
            continue
        shutil.copyfile(skl_fn, out_skl_fn)

    smpl_fns = glob(osp.join(root_dir, 'p*_a*/smpl_params.npz'))
    out_smpl_dir = osp.join(out_dir, 'smpl')
    os.makedirs(out_smpl_dir, exist_ok=True)
    for smpl_fn in tqdm(smpl_fns, desc='Copy smpl'):
        p = smpl_fn.split(os.sep)[-2]
        new_name = f'{p}_smpl_params.npz'
        out_smpl_fn = osp.join(out_smpl_dir, new_name)
        if os.path.exists(out_smpl_fn):
            continue
        shutil.copyfile(smpl_fn, out_smpl_fn)


def preprocess_humman_cropped(
    root_dir,
    out_dir,
    rgb_out_size=(224, 224),
    depth_out_size=(224, 224),
    num_workers=None,
    staging_dir=None,
    staging_chunk_size=5,
    yolo_conf=0.25,
    yolo_iou=0.45,
    yolo_device=None,
):
    if num_workers is None or num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    if staging_dir is None:
        _preprocess_humman_cropped_root(
            root_dir,
            out_dir,
            rgb_out_size,
            depth_out_size,
            num_workers,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_device=yolo_device,
        )
    else:
        seq_dirs = sorted(glob(osp.join(root_dir, 'p*_a*')))
        chunk_iter = _iter_chunks(seq_dirs, staging_chunk_size)
        for chunk_idx, chunk in enumerate(
            tqdm(chunk_iter, total=(len(seq_dirs) + staging_chunk_size - 1) // staging_chunk_size),
            start=1,
        ):
            stage_root = tempfile.mkdtemp(prefix=f"humman_stage_{chunk_idx}_", dir=staging_dir)
            _stage_sequence_dirs(chunk, stage_root)
            _preprocess_humman_cropped_root(
                stage_root,
                out_dir,
                rgb_out_size,
                depth_out_size,
                num_workers,
                yolo_conf=yolo_conf,
                yolo_iou=yolo_iou,
                yolo_device=yolo_device,
            )
            shutil.rmtree(stage_root, ignore_errors=True)
if __name__ == '__main__':
    # root_dir = '/data/shared/MMFi_Dataset'
    # rgb_dir = '/data/shared/MMFi_Defaced_RGB'
    # out_dir = 'data/mmfi'
    # rgb_out_size = (256, 192)
    # depth_out_size = (256, 192)

    # preprocess_mmfi(root_dir, rgb_dir, out_dir, rgb_out_size, depth_out_size, num_workers=16)

    root_dir = '/data/shared/humman_release_v1.0_point'
    out_dir = '/opt/data/humman_cropped'
    rgb_out_size = (224, 224)
    depth_out_size = (224, 224)
    # rgb_out_size = (320, 180)
    # depth_out_size = (320, 288)

    # preprocess_humman(root_dir, out_dir, rgb_out_size, depth_out_size, num_workers=16, staging_dir=None)
    # preprocess_humman(root_dir, out_dir, rgb_out_size, depth_out_size, num_workers=16, staging_dir='data/temp')
    preprocess_humman_cropped(root_dir, out_dir, rgb_out_size, depth_out_size, num_workers=16, staging_dir='data/temp')
