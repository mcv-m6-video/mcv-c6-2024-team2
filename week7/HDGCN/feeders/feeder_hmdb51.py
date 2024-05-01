import numpy as np
from copy import deepcopy

from torch.utils.data import Dataset

from feeders import tools
import pickle

class Feeder(Dataset):
    def __init__(self, data_path='hmdb51_2d_processed.pkl', p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False):
        """
        :param data_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        #  (np.ndarray, with shape [M x T x V x C]):
        #  The keypoint annotation. M: number of persons; T: number of frames (same as total_frames)
        # V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. );
        # C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).

        npz_data = []
        labels = []
        sample_name = []

        with open(self.data_path, "rb") as file:
            data = pickle.load(file)

        for k, v in data.items():
            keypoints = v['keypoint']
            label = v['label']

            for kp in keypoints:
                # Reshape back to M x T x V x C where M = 1
                kp_shape = kp.shape
                kp = kp.reshape(1, kp_shape[0], kp_shape[1], kp_shape[2])

                # M, T, V, C to C, T, V, M
                M, T, V, C = kp.shape
                kp = kp.reshape((M, T, V, C)).transpose(3, 1, 2, 0)

                npz_data.append(kp)
                labels.append(label)
                sample_name.append(k)

        self.data = npz_data
        self.label = labels
        self.sample_name = sample_name

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        clip_length = 5
        stride = 1
        data_numpy = tools.crop_resize(data_numpy, clip_length, stride)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import joint_pairs
            bone_data_numpy = np.zeros_like(data_numpy) # 3, T, V
            for v1, v2 in joint_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            data_numpy = bone_data_numpy
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
