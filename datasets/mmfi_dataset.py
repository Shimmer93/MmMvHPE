import os
import os.path as osp
import cv2
import numpy as np
import yaml
from typing import Callable, List, Optional, Sequence, Tuple, Union
from functools import lru_cache
from misc.timer import timer

from datasets.base_dataset import BaseDataset


def decode_config(config, split_to_use, protocol):
    assert split_to_use in config.keys(), f"split_to_use {split_to_use} not found in config."

    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                   'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    train_form = {}
    val_form = {}
    # Limitation to actions (protocol)
    if protocol == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif protocol == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions
    # Limitation to subjects and actions (split choices)
    if split_to_use == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(all_subjects))
            idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
            idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
            subjects_train = np.array(all_subjects)[idx_train].tolist()
            subjects_val = np.array(all_subjects)[idx_val].tolist()
            for subject in all_subjects:
                if subject in subjects_train:
                    if subject in train_form:
                        train_form[subject].append(action)
                    else:
                        train_form[subject] = [action]
                if subject in subjects_val:
                    if subject in val_form:
                        val_form[subject].append(action)
                    else:
                        val_form[subject] = [action]
            rs += 1
    elif split_to_use == 'cross_scene_split':
        subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                          'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                          'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
        subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif split_to_use == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    else:
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        actions_train = config['manual_split']['train_dataset']['actions']
        actions_val = config['manual_split']['val_dataset']['actions']
        for subject in subjects_train:
            train_form[subject] = actions_train
        for subject in subjects_val:
            val_form[subject] = actions_val

    dataset_config = {'train_dataset': train_form, 'val_dataset': val_form}
    return dataset_config

def get_scene(subject):
    if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
        return 'E01'
    elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
        return 'E02'
    elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
        return 'E03'
    elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
        return 'E04'
    else:
        raise ValueError('Subject does not exist in this dataset.')

class MMFiDataset(BaseDataset):
    def __init__(self, 
                 data_root: str,
                 rgb_root: Optional[str] = None,
                 split_config: str = 'configs/datasets/mmfi_split_config.yml',
                 split_to_use: str = 'random_split',
                 protocol: str = 'protocol1',
                 modality_names: Sequence[str] = ['rgb', 'depth', 'lidar', 'mmwave'],
                 seq_len: int = 5,
                 seq_step: int = 1,
                 mmwave_num_frames: int = 0,
                 lidar_num_frames: int = 0,
                 pad_seq: bool = False,
                 causal: bool = True,
                 pipeline: List[dict] = [],
                 test_mode: bool = False,
                 cache_ground_truth: bool = True):
        
        super().__init__(pipeline=pipeline)

        self.data_root = data_root
        self.rgb_root = rgb_root if rgb_root is not None else data_root
        
        # Load split config once
        with open(split_config, 'r') as f:
            split_config = yaml.safe_load(f)
        split_info = decode_config(split_config, split_to_use, protocol)
        self.split_info = split_info['val_dataset'] if test_mode else split_info['train_dataset']
        
        self.modality_names = modality_names
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.mmwave_num_frames = mmwave_num_frames
        self.lidar_num_frames = lidar_num_frames
        self.pad_seq = pad_seq
        self.causal = causal
        self.test_mode = test_mode
        self.cache_ground_truth = cache_ground_truth

        # Pre-load and cache data
        self.data_split = self.load_data_split()
        
        # Cache ground truth data to avoid repeated I/O
        if self.cache_ground_truth:
            self._gt_cache = {}
            self._preload_ground_truth()

    def _preload_ground_truth(self):
        """Pre-load all ground truth data into memory"""
        unique_gt_paths = set(item['gt_path'] for item in self.data_split)
        for gt_path in unique_gt_paths:
            self._gt_cache[gt_path] = np.load(gt_path)
                        
    def load_data_split(self):
        data_split = []
        
        # Pre-compute frame indices to avoid repeated calculation
        if self.causal:
            frame_offsets = list(range(-self.seq_len + 1, 1))
        else:
            half_len = (self.seq_len - 1) // 2
            frame_offsets = list(range(-half_len, half_len + 1))
        
        for subject, actions in self.split_info.items():
            scene = get_scene(subject)
            
            for action in actions:
                action_dir = osp.join(self.data_root, scene, subject, action)
                action_dir_rgb = osp.join(self.rgb_root, scene, subject, action)

                # Read frame lists once per action
                frame_lists = {}
                for modality in self.modality_names:
                    modality_path = osp.join(action_dir, modality)
                    if modality == 'rgb':
                        modality_path = osp.join(action_dir_rgb, modality)
                    frame_lists[modality] = sorted(os.listdir(modality_path))
                
                frame_list_ref = frame_lists[self.modality_names[0]]
                num_total_frames = len(frame_list_ref)

                if self.pad_seq:
                    start_idx = 0
                    num_frames = num_total_frames
                else:
                    start_idx = self.seq_len - 1 if self.causal else (self.seq_len - 1) // 2
                    num_frames = num_total_frames - (self.seq_len - 1) * self.seq_step

                gt_path = osp.join(action_dir, 'ground_truth.npy')

                # Pre-compute paths for all frames
                for idx in range(start_idx, num_frames):
                    frame_idx = int(frame_list_ref[idx].split('.')[0].split('frame')[1]) - 1
                    frame_idxs = [max(0, min(num_total_frames - 1, frame_idx + offset)) 
                                  for offset in frame_offsets]

                    data_dict = {
                        'modalities': self.modality_names,
                        'scene': scene,
                        'subject': subject,
                        'action': action,
                        'gt_path': gt_path,
                        'idx': frame_idx,
                        'frame_idxs': frame_idxs  # Store pre-computed indices
                    }

                    # Pre-compute all paths
                    for modality in self.modality_names:
                        if modality == 'rgb':
                            data_dict['rgb_paths'] = [
                                osp.join(action_dir_rgb, 'rgb', frame_lists['rgb'][i]) 
                                for i in frame_idxs
                            ]
                        elif modality == 'mmwave':
                            data_dict['mmwave_paths'] = self._compute_multi_frame_paths(
                                action_dir, modality, frame_lists[modality], 
                                frame_idxs, self.mmwave_num_frames
                            )
                        elif modality == 'lidar':
                            data_dict['lidar_paths'] = self._compute_multi_frame_paths(
                                action_dir, modality, frame_lists[modality], 
                                frame_idxs, self.lidar_num_frames
                            )
                        else:
                            data_dict[f'{modality}_paths'] = [
                                osp.join(action_dir, modality, frame_lists[modality][i]) 
                                for i in frame_idxs
                            ]

                    data_split.append(data_dict)

        return data_split
    
    def _compute_multi_frame_paths(self, action_dir, modality, frame_list, 
                                    frame_idxs, num_extra_frames):
        """Helper to compute multi-frame paths for lidar/mmwave"""
        multi_frame_paths = []
        max_idx = len(frame_list) - 1
        
        for i in frame_idxs:
            frame_paths = []
            for j in range(-num_extra_frames, num_extra_frames + 1):
                frame_idx = max(0, min(max_idx, i + j))
                frame_paths.append(osp.join(action_dir, modality, frame_list[frame_idx]))
            multi_frame_paths.append(frame_paths)
        
        return multi_frame_paths
    
    def read_frame(self, frame_path, modality):
        """Optimized frame reading with cv2 flags"""
        if modality == 'rgb':
            # Use cv2.IMREAD_COLOR for faster reading
            with timer('Read RGB frame'):
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif modality == 'depth':
            # Specify depth reading mode
            with timer('Read Depth frame'):
                frame = cv2.imread(frame_path)
        elif modality in ['lidar', 'mmwave']:
            # Use numpy's fromfile for faster binary reading
            dtype_size = 3 if modality == 'lidar' else 5
            with timer(f'Read {modality.upper()} frame'):
                frame = np.fromfile(frame_path, dtype=np.float64).reshape(-1, dtype_size)
        else:
            raise ValueError(f'Modality {modality} not supported.')
        return frame
    
    def __len__(self):
        return len(self.data_split)
    
    def __getitem__(self, idx):
        data_dict = self.data_split[idx]

        # Use cached ground truth if available
        if self.cache_ground_truth:
            gt_keypoints = self._gt_cache[data_dict['gt_path']][data_dict['idx']]
        else:
            gt_keypoints = np.load(data_dict['gt_path'])[data_dict['idx']]

        sample = {
            'gt_keypoints': gt_keypoints,
            'sample_id': f"{data_dict['scene']}_{data_dict['subject']}_{data_dict['action']}_{data_dict['idx']:03d}",
            'modalities': data_dict['modalities']
        }

        # Read frames for each modality
        for modality in data_dict['modalities']:
            frame_paths = data_dict[f'{modality}_paths']
            
            try:
                if modality in ['mmwave', 'lidar']:
                    # Vectorize concatenation
                    frames = []
                    for multi_frame_paths in frame_paths:
                        multi_frames = [self.read_frame(fp, modality) for fp in multi_frame_paths]
                        # Use vstack for better performance
                        frames.append(np.vstack(multi_frames))
                else:
                    frames = [self.read_frame(fp, modality) for fp in frame_paths]
                
                sample[f'input_{modality}'] = frames
            except Exception as e:
                print(f"Error reading {modality} frames for {sample['sample_id']}: {e}")
                # Add empty placeholder to avoid pipeline errors
                sample[f'input_{modality}'] = None

        sample = self.pipeline(sample)

        return sample
    
class MMFiPreprocessedDataset(MMFiDataset):

    def load_data_split(self):
        file_index = {m: {} for m in self.modality_names}
        for modality in self.modality_names:
            modality_path = osp.join(self.data_root, modality)
            all_files = sorted(os.listdir(modality_path))
            
            for fn in all_files:
                parts = fn.split('_')
                key = f'{parts[0]}_{parts[1]}_{parts[2]}'
                file_index[modality].setdefault(key, []).append(fn)

        for key in file_index[modality]:
            file_index[modality][key].sort()

        data_split = []
        # Pre-compute frame indices to avoid repeated calculation
        if self.causal:
            frame_offsets = list(range(-self.seq_len + 1, 1))
        else:
            half_len = (self.seq_len - 1) // 2
            frame_offsets = list(range(-half_len, half_len + 1))

        for subject, actions in self.split_info.items():
            scene = get_scene(subject)

            for action in actions:
                key = f"{scene}_{subject}_{action}"
                frame_lists = {}

                # Get pre-indexed filenames per modality
                for modality in self.modality_names:
                    frame_lists[modality] = file_index.get(modality, {}).get(key, [])

                frame_list_ref = frame_lists[self.modality_names[0]]
                num_total_frames = len(frame_list_ref)
                if num_total_frames == 0:
                    continue  # no data for this (scene, subject, action)

                if self.pad_seq:
                    start_idx = 0
                    num_frames = num_total_frames
                else:
                    start_idx = self.seq_len - 1 if self.causal else (self.seq_len - 1) // 2
                    num_frames = num_total_frames - (self.seq_len - 1) * self.seq_step

                gt_path = osp.join(self.data_root, 'gt', f"{key}.npy")

                # Pre-compute paths for all frames
                for idx in range(start_idx, num_frames):
                    # Use list index directly as frame index
                    frame_idx = idx
                    frame_idxs = [max(0, min(num_total_frames - 1, idx + offset))
                                  for offset in frame_offsets]

                    data_dict = {
                        'modalities': self.modality_names,
                        'scene': scene,
                        'subject': subject,
                        'action': action,
                        'gt_path': gt_path,
                        'idx': frame_idx,
                        'frame_idxs': frame_idxs,
                    }

                    # Pre-compute all paths
                    for modality in self.modality_names:
                        if modality == 'rgb':
                            data_dict['rgb_paths'] = [
                                osp.join(self.data_root, 'rgb', frame_lists['rgb'][i])
                                for i in frame_idxs
                            ]
                        elif modality == 'mmwave':
                            data_dict['mmwave_paths'] = self._compute_multi_frame_paths(
                                modality, frame_lists['mmwave'], frame_idxs, self.mmwave_num_frames
                            )
                        elif modality == 'lidar':
                            data_dict['lidar_paths'] = self._compute_multi_frame_paths(
                                modality, frame_lists['lidar'], frame_idxs, self.lidar_num_frames
                            )
                        else:
                            data_dict[f'{modality}_paths'] = [
                                osp.join(self.data_root, modality, frame_lists[modality][i])
                                for i in frame_idxs
                            ]

                    data_split.append(data_dict)

        return data_split

    def _compute_multi_frame_paths(self, modality, frame_list, frame_idxs, num_extra_frames):
        """Helper to compute multi-frame paths for lidar/mmwave"""
        multi_frame_paths = []
        max_idx = len(frame_list) - 1

        for i in frame_idxs:
            frame_paths = []
            for offset in range(-num_extra_frames, num_extra_frames + 1):
                frame_idx = max(0, min(max_idx, i + offset))
                frame_paths.append(osp.join(self.data_root, modality, frame_list[frame_idx]))
            multi_frame_paths.append(frame_paths)

        return multi_frame_paths

    def read_frame(self, frame_path, modality):
        """Optimized frame reading with cv2 flags"""
        if modality == 'rgb':
            # Use cv2.IMREAD_COLOR for faster reading
            # with timer('Read RGB frame'):
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif modality == 'depth':
            # Specify depth reading mode
            # with timer('Read Depth frame'):
            frame = cv2.imread(frame_path)
        elif modality in ['lidar', 'mmwave']:
            # Use numpy's fromfile for faster binary reading
            # with timer(f'Read {modality.upper()} frame'):
            frame = np.load(frame_path)
        else:
            raise ValueError(f'Modality {modality} not supported.')
        return frame