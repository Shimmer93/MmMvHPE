import numpy as np

JOINT_COLOR_MAP = [
    'black', 'red', 'brown', 'blue', 'deepskyblue', 'green', 'limegreen', 'orange', 'gold', 'purple', 'deeppink',
    'crimson', 'steelblue', 'darkviolet', 'slateblue', 'darkgoldenrod', 'turquoise', 'silver', 'salmon', 'limegreen',
    'pink', 'khaki', 'chocolate', 'cyan', 'olive'
]

def get_flip_indices(num_joints, left_indices, right_indices):
    flip_indices = []
    for i in range(num_joints):
        if i in left_indices:
            flip_indices.append(right_indices[left_indices.index(i)])
        elif i in right_indices:
            flip_indices.append(left_indices[right_indices.index(i)])
        else:
            flip_indices.append(i)
    
    return flip_indices
        
def get_left_right_bones(bones, left_indices, right_indices, flip_indices):
    left_bones = []
    right_bones = []
    for bone in bones:
        if bone[0] in left_indices and bone[1] not in right_indices:
            left_bones.append(bone)

    for left_bone in left_bones:
        right_bone = [flip_indices[left_bone[0]], flip_indices[left_bone[1]]]
        right_bones.append(right_bone)

    return left_bones, right_bones

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_adjacency_matrix(bones, num_joints):
    A = np.zeros((num_joints, num_joints))
    for bone in bones:
        A[bone[0], bone[1]] = 1
        A[bone[1], bone[0]] = 1
    for i in range(num_joints):
        A[i, i] = 1
    return A

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_adjacency_matrix(bones, num_joints):
    self_link = [(i, i) for i in range(num_joints)]
    inward = [(j, i) for (i, j) in bones]
    outward = [(i, j) for (i, j) in bones]
    A = get_spatial_graph(num_joints, self_link, inward, outward)
    return A

        
class COCOSkeleton:
    joint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle'
    ]
    bones = [
        [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11],
        [6, 12], [11, 13], [12, 14], [13, 15], [14, 16]
    ]
    left_indices = [1, 3, 5, 7, 9, 11, 13, 15]
    right_indices = [2, 4, 6, 8, 10, 12, 14, 16]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0
            
class SimpleCOCOSkeleton:
    joint_names = [
        'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
        'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    bones = [
        [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [1, 7],
        [2, 8], [7, 9], [8, 10], [9, 11], [10, 12]
    ]
    left_indices = [1, 3, 5, 7, 9, 11]
    right_indices = [2, 4, 6, 8, 10, 12]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0
    
class MMBodySkeleton:
    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", 
        "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    bones = [
        [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], 
        [9, 13], [9, 14], [12, 13], [12, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], 
        [18, 20], [19, 21]
    ]
    left_indices = [1, 3, 5, 7, 9, 11, 13, 15]
    right_indices = [2, 4, 6, 8, 10, 12, 14, 16]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

class H36MSkeleton:
    joint_names = [
        "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", 
        "waist", "neck", "nose", "head", "left_shoulder", "left_elbow", "left_wrist", 
        "right_shoulder", "right_elbow", "right_wrist"
    ]
    bones = [
        [0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [8, 11], [8, 14],
        [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]
    ]
    left_indices = [4, 5, 6, 11, 12, 13]
    right_indices = [1, 2, 3, 14, 15, 16]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

class MiliPointSkeleton:
    joint_names = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", 
        "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", 
        "right_eye", "left_eye", "right_ear", "left_ear"
    ]

    bones = [
        [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
        [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]
    ]
    left_indices = [2, 3, 4, 8, 9, 10, 14, 16]
    right_indices = [5, 6, 7, 11, 12, 13, 15, 17]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

class ITOPSkeleton:
    joint_names = [
        "nose", "neck", "right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_wrist", 
        "left_wrist", "spine", "right_hip", "left_hip", "right_knee", "left_knee", "right_ankle", 
        "left_ankle"
    ]
    bones = [
        [14, 12], [12, 10], [13, 11], [11, 9],  [10, 8], [9, 8], [8, 1], [1, 0], [7, 5], [5, 3], 
        [3, 1], [6, 4], [4, 2], [2, 1]
    ]
    left_indices = [3, 5, 7, 10, 12, 14]
    right_indices = [2, 4, 6, 9, 11, 13]

    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

class SMPLSkeleton:
    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", 
        "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", 
        "left_hand", "right_hand"
    ]
    bones = [
        [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11],
        [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21],
        [20, 22], [21, 23]
    ]
    left_indices = [1, 4, 7, 10, 13, 16, 18, 20, 22]
    right_indices = [2, 5, 8, 11, 14, 17, 19, 21, 23]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

def coco2simplecoco(joints):
    return joints[..., [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]

def mmbody2simplecoco(joints):
    return joints[..., [15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8], :]

def mmfi2simplecoco(joints):
    return joints[..., [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3], :]

def milipoint2simplecoco(joints):
    return joints[..., [0, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10], :]

def smpl2simplecoco(joints):
    return joints[..., [15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8], :]

def itop2simplecoco(joints):
    return joints[..., [0, 3, 2, 5, 4, 7, 6, 10, 9, 12, 11, 14, 13], :]

def mmfi2itop(joints):
    return joints[..., [9, 8, 14, 11, 15, 12, 16, 13, 7, 1, 4, 2, 5, 3, 6], :]

def mmbody2itop(joints):
    return joints[..., [15, 12, 17, 16, 19, 18, 21, 20, 6, 2, 1, 5, 4, 8, 7], :]