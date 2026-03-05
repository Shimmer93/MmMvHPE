import numpy as np
import torch

from misc import skeleton as skeleton_defs
from misc.utils import torch2numpy as to_numpy

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    # import numpy as np

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    # U,s,Vt = np.linalg.svd(A,full_matrices=False)
    try:
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
    except np.linalg.LinAlgError:
        print("SVD did not converge, using identity rotation")
        U = np.eye(A.shape[0])
        Vt = np.eye(A.shape[0])
        s = np.zeros(A.shape[0])

    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c

def mpjpe_func(preds, gts, reduce=True):
    N = preds.shape[0]
    num_joints = preds.shape[-2]

    mpjpe = np.sqrt(np.sum(np.square(preds - gts), axis=-1))
    if reduce: 
        mpjpe = np.mean(mpjpe)
    
    return mpjpe


def _validate_keypoints_shape(arr, name):
    if arr.ndim < 3 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., J, 3), got {arr.shape}.")


def _as_samples_joints3(arr):
    _validate_keypoints_shape(arr, "keypoints")
    num_joints = arr.shape[-2]
    return arr.reshape(-1, num_joints, 3), arr.shape


def _normalize_vec(v, eps=1e-6):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n <= eps):
        raise ValueError("Cannot normalize near-zero vector while estimating root rotation.")
    return v / n


def _estimate_root_rotation_from_keypoints(
    keypoints,
    neck_idx,
    bodycenter_idx,
    lhip_idx,
    rhip_idx,
):
    if keypoints.shape[-2] <= max(neck_idx, bodycenter_idx, lhip_idx, rhip_idx):
        raise ValueError(
            f"Need keypoints indices [{neck_idx},{bodycenter_idx},{lhip_idx},{rhip_idx}], got shape={keypoints.shape}."
        )

    neck = keypoints[..., neck_idx, :]
    body = keypoints[..., bodycenter_idx, :]
    lhip = keypoints[..., lhip_idx, :]
    rhip = keypoints[..., rhip_idx, :]

    x_axis = _normalize_vec(rhip - lhip)
    y_seed = _normalize_vec(neck - body)
    z_axis = _normalize_vec(np.cross(x_axis, y_seed))
    y_axis = _normalize_vec(np.cross(z_axis, x_axis))

    rot = np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)
    det = np.linalg.det(rot)
    if np.any(~np.isfinite(det)) or np.any(np.abs(det) < 1e-5):
        raise ValueError("Invalid root rotation matrix estimated from keypoints.")
    return rot


_SKELETON_CLASS_BY_KEY = {
    "smpl": skeleton_defs.SMPLSkeleton,
    "smplskeleton": skeleton_defs.SMPLSkeleton,
    "h36m": skeleton_defs.H36MSkeleton,
    "h36mskeleton": skeleton_defs.H36MSkeleton,
    "mmbody": skeleton_defs.MMBodySkeleton,
    "mmbodyskeleton": skeleton_defs.MMBodySkeleton,
    "panopticcoco19": skeleton_defs.PanopticCOCO19Skeleton,
    "panopticcoco19skeleton": skeleton_defs.PanopticCOCO19Skeleton,
    "coco": skeleton_defs.COCOSkeleton,
    "cocoskeleton": skeleton_defs.COCOSkeleton,
    "simplecoco": skeleton_defs.SimpleCOCOSkeleton,
    "simplecocoskeleton": skeleton_defs.SimpleCOCOSkeleton,
    "itop": skeleton_defs.ITOPSkeleton,
    "itopskeleton": skeleton_defs.ITOPSkeleton,
    "milipoint": skeleton_defs.MiliPointSkeleton,
    "milipointskeleton": skeleton_defs.MiliPointSkeleton,
}


def _canonical_key(name):
    return str(name).replace("_", "").replace("-", "").strip().lower()


def _get_skeleton_class(skeleton_name):
    key = _canonical_key(skeleton_name)
    if key not in _SKELETON_CLASS_BY_KEY:
        raise ValueError(
            f"Unsupported skeleton_name={skeleton_name}. "
            f"Supported: {sorted(_SKELETON_CLASS_BY_KEY.keys())}"
        )
    return _SKELETON_CLASS_BY_KEY[key]


def _first_joint_index(name_to_idx, candidates, field_name, skeleton_name):
    for name in candidates:
        idx = name_to_idx.get(name, None)
        if idx is not None:
            return idx
    raise ValueError(
        f"Cannot infer {field_name} index from skeleton_name={skeleton_name}. "
        f"Expected one of {candidates}, available joints={list(name_to_idx.keys())}."
    )


def _resolve_pcmpjpe_indices(
    skeleton_name,
    pelvis_idx=None,
    neck_idx=None,
    bodycenter_idx=None,
    lhip_idx=None,
    rhip_idx=None,
):
    skeleton_class = _get_skeleton_class(skeleton_name)
    joint_names = [str(x).lower() for x in skeleton_class.joint_names]
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    if pelvis_idx is None:
        pelvis_idx = int(getattr(skeleton_class, "center", -1))
        if pelvis_idx < 0:
            pelvis_idx = _first_joint_index(
                name_to_idx,
                ["pelvis", "mid_hip", "bodycenter", "body_center", "waist", "spine", "spine1"],
                "pelvis",
                skeleton_name,
            )
    if neck_idx is None:
        neck_idx = _first_joint_index(name_to_idx, ["neck"], "neck", skeleton_name)
    if bodycenter_idx is None:
        bodycenter_idx = _first_joint_index(
            name_to_idx,
            ["bodycenter", "body_center", "mid_hip", "pelvis", "waist", "spine", "spine1", "spine2"],
            "bodycenter",
            skeleton_name,
        )
    if lhip_idx is None:
        lhip_idx = _first_joint_index(name_to_idx, ["left_hip", "lhip"], "left_hip", skeleton_name)
    if rhip_idx is None:
        rhip_idx = _first_joint_index(name_to_idx, ["right_hip", "rhip"], "right_hip", skeleton_name)

    return int(pelvis_idx), int(neck_idx), int(bodycenter_idx), int(lhip_idx), int(rhip_idx)


def _resolve_pelvis_idx(skeleton_name, pelvis_idx=None):
    if pelvis_idx is not None:
        return int(pelvis_idx)
    skeleton_class = _get_skeleton_class(skeleton_name)
    pelvis_idx = int(getattr(skeleton_class, "center", -1))
    if pelvis_idx >= 0:
        return pelvis_idx
    joint_names = [str(x).lower() for x in skeleton_class.joint_names]
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    return int(
        _first_joint_index(
            name_to_idx,
            ["pelvis", "mid_hip", "bodycenter", "body_center", "waist", "spine", "spine1"],
            "pelvis",
            skeleton_name,
        )
    )


def _axis_angle_to_matrix(axis_angle, eps=1e-8):
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Axis-angle must end with dim=3, got {axis_angle.shape}.")
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = np.divide(axis_angle, theta, out=np.zeros_like(axis_angle), where=theta > eps)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    k = np.stack(
        [zero, -z, y,
         z, zero, -x,
         -y, x, zero],
        axis=-1,
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = np.eye(3, dtype=np.float32)
    eye = np.broadcast_to(eye, axis.shape[:-1] + (3, 3))
    theta = theta[..., None]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    kk = np.matmul(k, k)
    rot = eye + sin_t * k + (1.0 - cos_t) * kk
    near_zero = (np.squeeze(theta, axis=(-1, -2)) <= eps)[..., None, None]
    return np.where(near_zero, eye, rot).astype(np.float32)


def _to_rotmat(root_rot, name):
    root_rot = np.asarray(root_rot, dtype=np.float32)
    if root_rot.ndim < 2:
        raise ValueError(f"{name} must have shape (..., 3) or (..., 3, 3), got {root_rot.shape}.")
    if root_rot.shape[-2:] == (3, 3):
        return root_rot
    if root_rot.shape[-1] == 3:
        return _axis_angle_to_matrix(root_rot)
    raise ValueError(f"{name} must have shape (..., 3) or (..., 3, 3), got {root_rot.shape}.")


def pcmpjpe_func(
    preds,
    gts,
    pelvis_idx=0,
    neck_idx=None,
    bodycenter_idx=None,
    lhip_idx=None,
    rhip_idx=None,
    pred_root_rot=None,
    gt_root_rot=None,
    reduce=True,
):
    pred_samples, original_shape = _as_samples_joints3(np.asarray(preds, dtype=np.float32))
    gt_samples, _ = _as_samples_joints3(np.asarray(gts, dtype=np.float32))
    if pred_samples.shape != gt_samples.shape:
        raise ValueError(f"preds and gts shape mismatch: {pred_samples.shape} vs {gt_samples.shape}")
    if pelvis_idx < 0 or pelvis_idx >= pred_samples.shape[1]:
        raise ValueError(f"pelvis_idx out of range: {pelvis_idx} for {pred_samples.shape[1]} joints.")

    pred_centered = pred_samples - pred_samples[:, pelvis_idx:pelvis_idx + 1, :]
    gt_centered = gt_samples - gt_samples[:, pelvis_idx:pelvis_idx + 1, :]
    error = np.sqrt(np.sum(np.square(pred_centered - gt_centered), axis=-1))
    error = error.reshape(original_shape[:-1])
    return np.mean(error) if reduce else error

def pampjpe_func(preds, gts, reduce=True):
    N = preds.shape[0]
    num_joints = preds.shape[-2]

    pampjpe = np.zeros([N, num_joints])

    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]
        _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred = (b * frame_pred.dot(T)) + c
        pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))

    if reduce:
        pampjpe = np.mean(pampjpe)

    return pampjpe


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return to_numpy(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Cannot convert an empty list/tuple to numpy for MPJPE metrics.")
        items = []
        valid_shapes = []
        for item in x:
            if item is None:
                items.append(None)
                continue
            arr = _to_numpy(item)
            if not isinstance(arr, np.ndarray) or arr.ndim == 0 or arr.size == 0:
                items.append(None)
                continue
            items.append(arr)
            valid_shapes.append(arr.shape)

        if len(valid_shapes) == 0:
            raise ValueError("List/tuple has no valid array entries for MPJPE metrics.")
        first_shape = valid_shapes[0]
        if not all(shape == first_shape for shape in valid_shapes):
            shapes_all = [arr.shape if isinstance(arr, np.ndarray) else () for arr in items]
            raise ValueError(
                "Inconsistent list/tuple element shapes in MPJPE metrics: "
                f"{shapes_all}."
            )

        filled = []
        for arr in items:
            if arr is None:
                filled.append(np.full(first_shape, np.nan, dtype=np.float32))
            else:
                filled.append(arr.astype(np.float32, copy=False))
        return np.stack(filled, axis=0)
    return np.asarray(x)


def _select_keypoints_view(keypoints, *, field_name, sensor_idx=0, batch_size_hint=None):
    keypoints = _to_numpy(keypoints)

    if keypoints.ndim == 2:
        if keypoints.shape[-1] != 3:
            raise ValueError(f"`{field_name}` must end with xyz=3, got shape {keypoints.shape}.")
        return keypoints[None, ...]

    if keypoints.ndim == 3:
        if keypoints.shape[-1] != 3:
            raise ValueError(f"`{field_name}` must end with xyz=3, got shape {keypoints.shape}.")
        if batch_size_hint is None or keypoints.shape[0] == batch_size_hint:
            return keypoints
        if int(batch_size_hint) == 1:
            view_idx = min(int(sensor_idx), keypoints.shape[0] - 1)
            return keypoints[view_idx : view_idx + 1]
        raise ValueError(
            f"Ambiguous shape for `{field_name}`: {keypoints.shape} with batch_size_hint={batch_size_hint}."
        )

    if keypoints.ndim == 4:
        if keypoints.shape[-1] != 3:
            raise ValueError(f"`{field_name}` must end with xyz=3, got shape {keypoints.shape}.")
        if batch_size_hint is None:
            raise ValueError(
                f"`{field_name}` has shape {keypoints.shape}; batch_size_hint is required for 4D inputs."
            )
        if keypoints.shape[0] == batch_size_hint:
            view_idx = min(int(sensor_idx), keypoints.shape[1] - 1)
            return keypoints[:, view_idx]
        if int(batch_size_hint) == 1:
            view_idx = min(int(sensor_idx), keypoints.shape[0] - 1)
            return keypoints[view_idx : view_idx + 1]
        raise ValueError(
            f"Incompatible shape for `{field_name}`: {keypoints.shape} with batch_size_hint={batch_size_hint}."
        )

    raise ValueError(
        f"Unsupported shape for `{field_name}`: {keypoints.shape}. "
        "Expected [J,3], [B,J,3], [V,J,3], [B,V,J,3], or [V,B,J,3] with batch hint."
    )


def _resolve_keypoints(preds, targets, *, pred_key, target_key, sensor_idx=0):
    if pred_key not in preds:
        raise KeyError(f"Missing prediction key `{pred_key}` for MPJPE metric.")
    if target_key not in targets:
        raise KeyError(f"Missing target key `{target_key}` for MPJPE metric.")

    pred_raw = preds[pred_key]
    target_raw = targets[target_key]

    target_batch_hint = None
    target_arr = _to_numpy(target_raw)
    if target_arr.ndim >= 3:
        target_batch_hint = target_arr.shape[0]

    pred_keypoints = _select_keypoints_view(
        pred_raw,
        field_name=pred_key,
        sensor_idx=sensor_idx,
        batch_size_hint=target_batch_hint,
    )
    target_keypoints = _select_keypoints_view(
        target_raw,
        field_name=target_key,
        sensor_idx=sensor_idx,
        batch_size_hint=pred_keypoints.shape[0],
    )

    if pred_keypoints.shape[0] != target_keypoints.shape[0]:
        pred_keypoints = _select_keypoints_view(
            pred_raw,
            field_name=pred_key,
            sensor_idx=sensor_idx,
            batch_size_hint=target_keypoints.shape[0],
        )

    if pred_keypoints.shape != target_keypoints.shape:
        raise ValueError(
            f"Prediction/target shape mismatch for keys `{pred_key}` and `{target_key}`: "
            f"{pred_keypoints.shape} vs {target_keypoints.shape}."
        )

    pred_valid = np.isfinite(pred_keypoints).reshape(pred_keypoints.shape[0], -1).all(axis=1)
    target_valid = np.isfinite(target_keypoints).reshape(target_keypoints.shape[0], -1).all(axis=1)
    valid = pred_valid & target_valid
    pred_keypoints = pred_keypoints[valid]
    target_keypoints = target_keypoints[valid]

    return pred_keypoints, target_keypoints


class MPJPE:
    def __init__(
        self,
        affix=None,
        use_smpl=False,
        root_joint_idx=-1,
        pred_key=None,
        target_key='gt_keypoints',
        sensor_idx=0,
    ):
        self.affix = affix
        self.use_smpl = use_smpl
        self.root_joint_idx = root_joint_idx
        self.pred_key = pred_key
        self.target_key = target_key
        self.sensor_idx = int(sensor_idx)
        if self.sensor_idx < 0:
            raise ValueError(f"sensor_idx must be >= 0, got {self.sensor_idx}.")
        self.name = f'mpjpe_{affix}' if affix is not None else 'mpjpe'

    def __call__(self, preds, targets):
        default_pred_key = 'pred_smpl_keypoints' if self.use_smpl else 'pred_keypoints'
        pred_key = self.pred_key or default_pred_key
        pred_keypoints, target_keypoints = _resolve_keypoints(
            preds,
            targets,
            pred_key=pred_key,
            target_key=self.target_key,
            sensor_idx=self.sensor_idx,
        )
        if pred_keypoints.shape[0] == 0:
            return np.nan

        if self.root_joint_idx >= 0:
            pred_keypoints = pred_keypoints - pred_keypoints[..., self.root_joint_idx:self.root_joint_idx+1, :]
            target_keypoints = target_keypoints - target_keypoints[..., self.root_joint_idx:self.root_joint_idx+1, :]

        mpjpe = mpjpe_func(pred_keypoints, target_keypoints, reduce=True)
        return mpjpe

class PAMPJPE:
    def __init__(
        self,
        affix=None,
        use_smpl=False,
        root_joint_idx=-1,
        pred_key=None,
        target_key='gt_keypoints',
        sensor_idx=0,
    ):
        self.affix = affix
        self.use_smpl = use_smpl
        self.root_joint_idx = root_joint_idx
        self.pred_key = pred_key
        self.target_key = target_key
        self.sensor_idx = int(sensor_idx)
        if self.sensor_idx < 0:
            raise ValueError(f"sensor_idx must be >= 0, got {self.sensor_idx}.")
        self.name = f'pampjpe_{affix}' if affix is not None else 'pampjpe'

    def __call__(self, preds, targets):
        default_pred_key = 'pred_smpl_keypoints' if self.use_smpl else 'pred_keypoints'
        pred_key = self.pred_key or default_pred_key
        pred_keypoints, target_keypoints = _resolve_keypoints(
            preds,
            targets,
            pred_key=pred_key,
            target_key=self.target_key,
            sensor_idx=self.sensor_idx,
        )
        if pred_keypoints.shape[0] == 0:
            return np.nan

        if self.root_joint_idx >= 0:
            pred_keypoints = pred_keypoints - pred_keypoints[..., self.root_joint_idx:self.root_joint_idx+1, :]
            target_keypoints = target_keypoints - target_keypoints[..., self.root_joint_idx:self.root_joint_idx+1, :]

        pampjpe = pampjpe_func(pred_keypoints, target_keypoints, reduce=True)
        return pampjpe


class PCMPJPE:
    def __init__(
        self,
        affix=None,
        use_smpl=False,
        skeleton_name="smpl",
        pelvis_idx=None,
        neck_idx=None,
        bodycenter_idx=None,
        lhip_idx=None,
        rhip_idx=None,
        pred_key=None,
        target_key='gt_keypoints',
        sensor_idx=0,
    ):
        self.affix = affix
        self.use_smpl = use_smpl
        self.skeleton_name = skeleton_name
        self.pelvis_idx = pelvis_idx
        self.neck_idx = neck_idx
        self.bodycenter_idx = bodycenter_idx
        self.lhip_idx = lhip_idx
        self.rhip_idx = rhip_idx
        self.pred_key = pred_key
        self.target_key = target_key
        self.sensor_idx = int(sensor_idx)
        if self.sensor_idx < 0:
            raise ValueError(f"sensor_idx must be >= 0, got {self.sensor_idx}.")
        self.name = f'pcmpjpe_{affix}' if affix is not None else 'pcmpjpe'

    def __call__(self, preds, targets):
        default_pred_key = 'pred_smpl_keypoints' if self.use_smpl else 'pred_keypoints'
        pred_key = self.pred_key or default_pred_key
        pred_keypoints, target_keypoints = _resolve_keypoints(
            preds,
            targets,
            pred_key=pred_key,
            target_key=self.target_key,
            sensor_idx=self.sensor_idx,
        )
        if pred_keypoints.shape[0] == 0:
            return np.nan

        pelvis_idx = _resolve_pelvis_idx(self.skeleton_name, self.pelvis_idx)

        pcmpjpe = pcmpjpe_func(
            pred_keypoints,
            target_keypoints,
            pelvis_idx=pelvis_idx,
            reduce=True,
        )
        return pcmpjpe
