import yaml
from argparse import Namespace
from collections import OrderedDict
import time
from contextlib import contextmanager
import pickle

def load_cfg(cfg):
    hyp = None
    if isinstance(cfg, str):
        with open(cfg, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    return Namespace(**hyp)

def merge_args_cfg(args, cfg):
    dict0 = vars(args)
    dict1 = vars(cfg)
    dict = {**dict0, **dict1}

    return Namespace(**dict)

def torch2numpy(tensor):
    return tensor.detach().cpu().numpy()

def delete_prefix_from_state_dict(state_dict, prefix):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_state_dict_part(model, state_dict, prefix):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
    model.load_state_dict(new_state_dict, strict=False)

def exists_and_is_true(hparams, key):
    return hasattr(hparams, key) and getattr(hparams, key)

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f} ms")

def load(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def dump(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)