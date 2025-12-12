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
from models import SMPL
from misc.utils import load, dump

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize SMPL parameters for given keypoints')
    parser.add_argument('--dataset_name', '-n', dest='dataset_name',
                        help='select dataset',
                        default='mmfi', type=str)
    parser.add_argument('--preds_path', dest='preds_path',
                        help='path of predictions',
                        default=None, type=str)
    parser.add_argument('--model_path', dest='model_path',
                        help='path of smpl model',
                        default='weights/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl', type=str)
    parser.add_argument('--joint_regressor_extra_path', dest='joint_regressor_extra_path',
                        help='path of extra joint regressor',
                        default=None, type=str)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='batch size for fitting',
                        default=1024, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate',
                        help='learning rate for optimizer',
                        default=0.02, type=float)
    parser.add_argument('--optimize_shape', dest='optimize_shape',
                        help='whether to optimize shape parameters',
                        action='store_true')
    parser.add_argument('--optimize_scale', dest='optimize_scale',
                        help='whether to optimize scale parameter',
                        action='store_true')
    parser.add_argument('--max_epoch', dest='max_epoch',
                        help='maximum number of epochs for optimization',
                        default=1000, type=int)
    parser.add_argument('--patience', dest='patience',
                        help='patience for early stopping',
                        default=10, type=int)
    parser.add_argument('--min_delta', dest='min_delta',
                        help='minimum change in loss to qualify as improvement',
                        default=5e-4, type=float)
    parser.add_argument('--one_batch', dest='one_batch',
                        help='whether to fit only one batch for testing',
                        action='store_true')
    args = parser.parse_args()
    return args

def easy_track(iterable, description=""):
    return track(iterable, description=description, complete_style="dim cyan", total=len(iterable))

def init_params(args, batch_size, device):
    params = {}
    params["pose_params"] = torch.zeros(batch_size, 72).to(device)
    params["shape_params"] = torch.zeros(batch_size, 10).to(device)
    params["scale"] = torch.ones([1]).to(device)
    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(args.optimize_shape)
    params["scale"].requires_grad = bool(args.optimize_scale)
    return params

def init_optimizer(args, params):
    optim_params = [{'params': params["pose_params"], 'lr': args.learning_rate},
                    {'params': params["shape_params"], 'lr': args.learning_rate},
                    {'params': params["scale"], 'lr': args.learning_rate*10},]
    optimizer = torch.optim.Adam(optim_params)
    return optimizer

def init_model(args, device):
    smpl_model = SMPL(center_idx=0, model_path=args.model_path)
    smpl_model = smpl_model.to(device)
    return smpl_model

def get_index_map(dataset_name):
    match dataset_name:
        case 'mmfi' | 'mmfi_preproc':
            index_map = [
                [0, 0], [2, 1], [5, 2], [8, 3],
                [1, 4], [4, 5], [7, 6], [12, 8],
                [13, 11], [18, 12], [20, 13],
                [14, 14], [19, 15], [21, 16]
            ]
        case _:
            raise NotImplementedError(f'Unknown dataset: {dataset_name}')
    return index_map

def index_map_to_selected_indices(index_map, device):
    smpl_indices = []
    dataset_indices = []
    for p in index_map:
        smpl_indices.append(p[0])
        dataset_indices.append(p[1])
    smpl_indices = torch.tensor(smpl_indices).to(device)
    dataset_indices = torch.tensor(dataset_indices).to(device)

    return smpl_indices, dataset_indices

class Meters:
    def __init__(self, eps=-1e-3, stop_threshold=10) -> None:
        self.eps = eps
        self.stop_threshold = stop_threshold
        self.avg = 0
        self.cnt = 0
        self.reset_early_stop()

    def reset_early_stop(self):
        self.min_loss = float('inf')
        self.satis_num = 0
        self.update_res = True
        self.early_stop = False

    def update_avg(self, val, k=1):
        self.avg = self.avg + (val - self.avg) * k / (self.cnt + k)
        self.cnt += k

    def update_early_stop(self, val):
        delta = (val - self.min_loss) / self.min_loss
        if float(val) < self.min_loss:
            self.min_loss = float(val)
            self.update_res = True
        else:
            self.update_res = False
        self.satis_num = self.satis_num + 1 if delta >= self.eps else 0
        self.early_stop = self.satis_num >= self.stop_threshold
    
rotate = {
    'HumanAct12': [1., -1., -1.],
    'CMU_Mocap': [0.05, 0.05, 0.05],
    'UTD_MHAD': [-1., 1., -1.],
    'Human3.6M': [-0.001, -0.001, 0.001],
    'mmfi': [1., -1., -1.],
    'NTU': [1., 1., -1.],
    'HAA4D': [1., -1., -1.],
}

def transform(name, arr: np.ndarray):
    arr_new = arr.copy()
    rot = np.array(rotate[name])
    if name == 'mmfi':
        arr_new[:, 0, :] -= np.array([0., 0.125, 0.])
    center = arr_new[:, 0, :].copy()
    arr_new -= center[:, np.newaxis, :]
    arr_new *= rot[np.newaxis, np.newaxis, :]
    return arr_new, rot, center

def train(args, model, kps2fit, device):
    params = init_params(args, kps2fit.shape[0], device)
    optimizer = init_optimizer(args, params)
    index_map = get_index_map(args.dataset_name)
    smpl_indices, dataset_indices = index_map_to_selected_indices(index_map, device)

    pose = params["pose_params"]
    beta = params["shape_params"]
    scale = params["scale"]
    kps2fit = kps2fit.to(device)

    meter = Meters(stop_threshold=args.patience, eps=-args.min_delta)

    for epoch in easy_track(range(args.max_epoch), description="Fitting SMPL model"):
        _, kps = model(pose, beta)
        loss = F.smooth_l1_loss(scale * kps.index_select(1, smpl_indices),
                                kps2fit.index_select(1, dataset_indices))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter.update_early_stop(loss.item())
        if meter.update_res:
            res = params, kps, model
        if meter.early_stop:
            print(f"Early stopping at epoch {epoch+1} with loss {loss.item():.6f}, best loss {meter.min_loss:.6f}")
            break

    print(f"Optimization finished, final loss: {loss.item():.6f}, best loss: {meter.min_loss:.6f}")

    return res

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = load(args.preds_path)
    outputs = {}

    model = init_model(args, device)

    for prefix in ['pred_', 'gt_']:
        key = prefix + 'keypoints'
        if preds[key] is not None:
            preds_key, rot, center = transform(args.dataset_name, deepcopy(preds[key]))
            preds_key = torch.from_numpy(preds_key).float().to(device)
            num_samples = preds_key.shape[0]

            outputs[prefix + 'pose'] = []
            outputs[prefix + 'beta'] = []
            for i in range(0, num_samples, args.batch_size):
                kps2fit = preds_key[i:i+args.batch_size]
                params, kps, model = train(args, model, kps2fit, device)
                outputs[prefix + 'pose'].append(params["pose_params"].detach().cpu().numpy())
                outputs[prefix + 'beta'].append(params["shape_params"].detach().cpu().numpy())
                if args.one_batch:
                    break
            outputs[prefix + 'pose'] = np.concatenate(outputs[prefix + 'pose'], axis=0)
            outputs[prefix + 'beta'] = np.concatenate(outputs[prefix + 'beta'], axis=0)
            outputs[prefix + 'rot'] = rot
            if args.one_batch:
                outputs[prefix + 'center'] = center[:args.batch_size]
            else:
                outputs[prefix + 'center'] = center

    dump(outputs, args.preds_path.replace('.pkl', '_smpl2.pkl'))
    print("Fitting finished!")

            

    
    