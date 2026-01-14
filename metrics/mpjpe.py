import numpy as np

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

class MPJPE:
    def __init__(self, affix=None, use_smpl=False, root_joint_idx=-1):
        self.affix = affix
        self.use_smpl = use_smpl
        self.root_joint_idx = root_joint_idx
        self.name = f'mpjpe_{affix}' if affix is not None else 'mpjpe'

    def __call__(self, preds, targets):
        if self.use_smpl:
            pred_keypoints = to_numpy(preds['pred_smpl_keypoints'])
        else:
            pred_keypoints = to_numpy(preds['pred_keypoints'])
        target_keypoints = to_numpy(targets['gt_keypoints'])

        if self.root_joint_idx >= 0:
            pred_keypoints = pred_keypoints - pred_keypoints[..., self.root_joint_idx:self.root_joint_idx+1, :]
            target_keypoints = target_keypoints - target_keypoints[..., self.root_joint_idx:self.root_joint_idx+1, :]

        mpjpe = mpjpe_func(pred_keypoints, target_keypoints, reduce=True)
        return mpjpe

class PAMPJPE:
    def __init__(self, affix=None, use_smpl=False, root_joint_idx=-1):
        self.affix = affix
        self.use_smpl = use_smpl
        self.root_joint_idx = root_joint_idx
        self.name = f'pampjpe_{affix}' if affix is not None else 'pampjpe'

    def __call__(self, preds, targets):
        if self.use_smpl:
            pred_keypoints = to_numpy(preds['pred_smpl_keypoints'])
        else:
            pred_keypoints = to_numpy(preds['pred_keypoints'])
        target_keypoints = to_numpy(targets['gt_keypoints'])

        if self.root_joint_idx >= 0:
            pred_keypoints = pred_keypoints - pred_keypoints[..., self.root_joint_idx:self.root_joint_idx+1, :]
            target_keypoints = target_keypoints - target_keypoints[..., self.root_joint_idx:self.root_joint_idx+1, :]

        pampjpe = pampjpe_func(pred_keypoints, target_keypoints, reduce=True)
        return pampjpe