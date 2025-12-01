import os
import os.path as osp
import cv2
import numpy as np
import yaml
from typing import Callable, List, Optional, Sequence, Tuple, Union

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
                 test_mode: bool = False):
        
        super().__init__(pipeline=pipeline)

        self.data_root = data_root
        self.rgb_root = rgb_root if rgb_root is not None else data_root
        with open(split_config, 'r') as f:
            split_config = yaml.safe_load(f)
        split_info = decode_config(split_config, split_to_use, protocol)
        if test_mode:
            self.split_info = split_info['val_dataset']
        else:
            self.split_info = split_info['train_dataset']
        self.modality_names = modality_names
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.mmwave_num_frames = mmwave_num_frames
        self.lidar_num_frames = lidar_num_frames
        self.pad_seq = pad_seq
        self.causal = causal
        self.test_mode = test_mode

        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_data_paths()

        self.data_split = self.load_data_split()

    def load_data_paths(self):
        for scene in sorted(os.listdir(self.data_root)):
            self.scenes[scene] = {}

            for subject in sorted(os.listdir(osp.join(self.data_root, scene))):
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}

                for action in sorted(os.listdir(osp.join(self.data_root, scene, subject))):
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    
                    for modality in self.modality_names:
                        data_path = osp.join(self.data_root, scene, subject, action, modality)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path
                        
    def load_data_split(self):
        data_split = tuple()
        
        for subject, actions in self.split_info.items():
            
            for action in actions:
                action_dir = osp.join(self.data_root, get_scene(subject), subject, action)
                action_dir_rgb = osp.join(self.rgb_root, get_scene(subject), subject, action)

                frame_lists = {modality: sorted(os.listdir(osp.join(action_dir, modality))) for modality in self.modality_names}
                frame_list_ref = frame_lists[self.modality_names[0]]
                frame_list_rgb = sorted(os.listdir(osp.join(action_dir_rgb, 'rgb')))

                if self.pad_seq:
                    start_idx = 0
                    num_frames = len(frame_list_ref)
                else:
                    start_idx = self.seq_len - 1 if self.causal else (self.seq_len - 1) // 2
                    num_frames = len(frame_list_ref) - (self.seq_len - 1) * self.seq_step

                for idx in range(start_idx, num_frames):
                    frame_idx = int(frame_list_ref[idx].split('.')[0].split('frame')[1]) - 1
                    if self.causal:
                        frame_idxs = [frame_idx - self.seq_len + 1 + i for i in range(self.seq_len)]
                    else:
                        half_len = (self.seq_len - 1) // 2
                        frame_idxs = [frame_idx - half_len + i for i in range(self.seq_len)]

                    data_dict = {
                        'modalities': self.modality_names,
                        'scene': get_scene(subject),
                        'subject': subject,
                        'action': action,
                        'gt_path': osp.join(action_dir, 'ground_truth.npy'),
                        'idx': frame_idx
                    }

                    for modality in self.modality_names:
                        if modality == 'rgb':
                            data_dict['rgb_paths'] = [osp.join(action_dir_rgb, 'rgb', frame_list_rgb[i]) for i in frame_idxs]
                        elif modality == 'mmwave':
                            data_dict['mmwave_paths'] = []
                            for i in frame_idxs:
                                mmwave_frame_paths = []
                                for j in range(2 * self.mmwave_num_frames + 1):
                                    mmwave_frame_idx = max(0, min(len(frame_lists[modality]) - 1, i - self.mmwave_num_frames + j))
                                    mmwave_frame_paths.append(osp.join(action_dir, modality, frame_lists[modality][mmwave_frame_idx]))
                                data_dict['mmwave_paths'].append(mmwave_frame_paths)
                        elif modality == 'lidar':
                            data_dict['lidar_paths'] = []
                            for i in frame_idxs:
                                lidar_frame_paths = []
                                for j in range(2 * self.lidar_num_frames + 1):
                                    lidar_frame_idx = max(0, min(len(frame_lists[modality]) - 1, i - self.lidar_num_frames + j))
                                    lidar_frame_paths.append(osp.join(action_dir, modality, frame_lists[modality][lidar_frame_idx]))
                                data_dict['lidar_paths'].append(lidar_frame_paths)
                        else:
                            data_dict[f'{modality}_paths'] = [osp.join(action_dir, modality, frame_lists[modality][i]) for i in frame_idxs]

                    data_split += (data_dict, )

        return data_split
    
    def read_frame(self, frame_path, modality):
        if modality == 'rgb':
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif modality == 'depth':
            frame = cv2.imread(frame_path) # TODO: depth image reading
        elif modality == 'lidar':
            with open(frame_path, 'rb') as f:
                raw_frame = f.read()
            frame = np.frombuffer(raw_frame, dtype=np.float64).reshape(-1, 3)
        elif modality == 'mmwave':
            with open(frame_path, 'rb') as f:
                raw_frame = f.read()
            frame = np.frombuffer(raw_frame, dtype=np.float64).reshape(-1, 5)
        else:
            raise ValueError(f'Modality {modality} not supported.')
        return frame
    
    def __len__(self):
        return len(self.data_split)
    
    def __getitem__(self, idx):
        data_dict = self.data_split[idx]

        gt_keypoints = np.load(data_dict['gt_path'])[data_dict['idx'], ...]

        sample = {}
        sample['gt_keypoints'] = gt_keypoints
        sample['sample_id'] = f"{data_dict['scene']}_{data_dict['subject']}_{data_dict['action']}_{data_dict['idx']:03d}"
        sample['modalities'] = data_dict['modalities']

        for modality in data_dict['modalities']:
            frame_paths = data_dict[f'{modality}_paths']
            try:
                if modality in ['mmwave', 'lidar']:
                    frames = []
                    for multi_frame_paths in frame_paths:
                        multi_frames = []
                        for frame_path in multi_frame_paths:
                            frame = self.read_frame(frame_path, modality)
                            multi_frames.append(frame)
                        multi_frames = np.concatenate(multi_frames, axis=0)
                        frames.append(multi_frames)
                else:
                    frames = [self.read_frame(frame_path, modality) for frame_path in frame_paths]
                sample[f'input_{modality}'] = frames
            except Exception as e:
                print(f"Error reading {modality} frames: {e}")

        sample = self.pipeline(sample)

        return sample