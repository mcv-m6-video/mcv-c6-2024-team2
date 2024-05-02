""" Dataset class for HMDB51 dataset. """

import os
import random
from enum import Enum
import pickle
import numpy as np
from PIL import Image

import tools

from glob import glob, escape
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
# from torchvision.transforms import v2
import torchvision.transforms as transforms
from torchvision.transforms import functional as F



class HMDB51Dataset(Dataset):
    """
    Dataset class for HMDB51 dataset.
    """

    class Split(Enum):
        """
        Enum class for dataset splits.
        """

        TEST_ON_SPLIT_1 = 1
        TEST_ON_SPLIT_2 = 2
        TEST_ON_SPLIT_3 = 3

    class Regime(Enum):
        """
        Enum class for dataset regimes.
        """

        TRAINING = 1
        TESTING = 2
        VALIDATION = 3

    CLASS_NAMES = [
        "brush_hair",
        "catch",
        "clap",
        "climb_stairs",
        "draw_sword",
        "drink",
        "fall_floor",
        "flic_flac",
        "handstand",
        "hug",
        "kick",
        "kiss",
        "pick",
        "pullup",
        "push",
        "ride_bike",
        "run",
        "shoot_ball",
        "shoot_gun",
        "situp",
        "smoke",
        "stand",
        "sword",
        "talk",
        "turn",
        "wave",
        "cartwheel",
        "chew",
        "climb",
        "dive",
        "dribble",
        "eat",
        "fencing",
        "golf",
        "hit",
        "jump",
        "kick_ball",
        "laugh",
        "pour",
        "punch",
        "pushup",
        "ride_horse",
        "shake_hands",
        "shoot_bow",
        "sit",
        "smile",
        "somersault",
        "swing_baseball",
        "sword_exercise",
        "throw",
        "walk",
    ]

    def __init__(
        self,
        videos_dir: str,
        annotations_dir: str,
        split: Split,
        regime: Regime,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        skeleton_file: str = os.path.join('data', 'hmdb51_2d_processed.pkl')
    ) -> None:
        """
        Initialize HMDB51 dataset.

        Args:
            videos_dir (str): Directory containing video files.
            annotations_dir (str): Directory containing annotation files.
            split (Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            regime (Regimes): Dataset regime (TRAINING, TESTING, VALIDATION).
            split (Splits): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            clip_length (int): Number of frames of the clips.
            crop_size (int): Size of spatial crops (squares).
            temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        """
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride
        self.skeleton_file = skeleton_file

        self.annotation_skel = self._read_annotation_skeleton()
        self.annotation_video = self._read_annotation_video()
        self.transform = self._create_transform()

    def _read_annotation_skeleton(self):
        with open(self.skeleton_file, "rb") as file:
            data = pickle.load(file)

        return data

    def _read_annotation_video(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: Dataframe containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in HMDB51Dataset.CLASS_NAMES:
            annotation_file = os.path.join(
                self.annotations_dir, class_name + split_suffix
            )
            df = pd.read_csv(annotation_file, sep=" ").dropna(
                axis=1, how="all"
            )  # drop empty columns
            df.columns = ["video_name", "train_or_test"]
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={"video_name": "video_path"})
            df["video_path"] = os.path.join(self.videos_dir, class_name, "") + df[
                "video_path"
            ].replace("\.avi$", "", regex=True)
            df = df.rename(columns={"train_or_test": "class_id"})
            df["class_id"] = HMDB51Dataset.CLASS_NAMES.index(class_name)
            annotation += [df]

        return pd.concat(annotation, ignore_index=True)
    
    def _read_image(self, file_path):
        # Open the image file
        with Image.open(file_path) as img:
            # Convert the image to RGB if it's not in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert the image to a numpy array
            img_np = np.array(img)
            
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            img_tensor = self.transform(img_tensor)
            return img_tensor



    def _create_transform(self) -> transforms.Compose:
        """
        Create transform based on the dataset regime.

        Returns:
            v2.Compose: Transform for the dataset.
        """
        if self.regime == HMDB51Dataset.Regime.TRAINING:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    self.crop_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ToTensor(),  # Convert to torch tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(self.crop_size * 1.15)),  # Slightly larger than crop size
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def get_num_classes(self) -> int:
        """
        Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(HMDB51Dataset.CLASS_NAMES)

    def __len__(self) -> int:
        """
        Get the length (number of videos) of the dataset.

        Returns:
            int: Length (number of videos) of the dataset.
        """
        return len(self.annotation_video)
    
    def _get_keypoints(self, video_name):
        # Get just the first skeleton - the one with the highest confidence
        keypoints = self.annotation_skel[video_name]['keypoint']
        print("label in kp ", self.annotation_skel[video_name]['label'])
        label = self.annotation_skel[video_name]['label']
        kp = keypoints[0]

        # Reshape back to M x T x V x C where M = 1
        kp_shape = kp.shape
        kp = kp.reshape(1, kp_shape[0], kp_shape[1], kp_shape[2])

        # M, T, V, C to C, T, V, M
        M, T, V, C = kp.shape
        kp = kp.reshape((M, T, V, C)).transpose(3, 1, 2, 0)

        clip_length = 5
        stride = 1
        kp = tools.crop_resize(kp, clip_length, stride)

        # from .bone_pairs import joint_pairs
        # bone_kp = np.zeros_like(kp) # 3, T, V
        # for v1, v2 in joint_pairs:
        #     bone_kp[:, :, v1] = kp[:, :, v1] - kp[:, :, v2]
        # kp = bone_kp

        return kp, label

    def __getitem__(self, idx: int) -> tuple:
        """
        Get item (video) from the dataset.

        Args:
            idx (int): Index of the item (video).

        Returns:
            tuple: Tuple containing video, label, and video path.
        """
        df_idx = self.annotation_video.iloc[idx]

        # Get video path from the annotation dataframe and check if it exists
        video_path = df_idx["video_path"]
        assert os.path.exists(video_path)

        # Read frames' paths from the video
        frame_paths = sorted(
            glob(os.path.join(escape(video_path), "*.jpg"))
        )  # get sorted frame paths
        video_len = len(frame_paths)

        # Get name of the video to retrieve keypoints
        video_name = video_path.split('/')[-1]
        kp, label = self._get_keypoints(video_name)

        if video_len <= self.clip_length * self.temporal_stride:
            # Not enough frames to create the clip
            clip_begin, clip_end = 0, video_len
        else:
            # Randomly select a clip from the video with the desired length (start and end frames are inclusive)
            clip_begin = random.randint(
                0, max(video_len - self.clip_length * self.temporal_stride, 0)
            )
            clip_end = clip_begin + self.clip_length * self.temporal_stride

        # Read frames from the video with the desired temporal subsampling
        video = None
        for i, path in enumerate(
            frame_paths[clip_begin : clip_end : self.temporal_stride]
        ):
            frame = self._read_image(path)  # (C, H, W)
            if video is None:
                video = torch.zeros(
                    (self.clip_length, 3, frame.shape[1], frame.shape[2]),
                    dtype=torch.uint8,
                )
            video[i] = frame

        # Get label from the annotation dataframe and make sure video was read
        # label = df_idx["class_id"]
        assert video is not None

        print("label in getitem ", label)

        return video, kp, label, video_path
    
    def collate_fn(self, batch: list) -> dict:
        """
        Collate function for creating batches.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """
        # [(clip1, label1, path1), (clip2, label2, path2), ...]
        #   -> ([clip1, clip2, ...], [label1, label2, ...], [path1, path2, ...])
        unbatched_clips, unbatched_keypoints, unbatched_labels, paths = zip(*batch)

        # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
        transformed_clips = [
            clip.permute(1, 0, 2, 3) for clip in unbatched_clips
        ]

        # Concatenate clips along the batch dimension:
        # B * [(C, T, H, W)] -> B * [(1, C, T, H, W)] -> (B, C, T, H, W)
        batched_clips = torch.cat([d.unsqueeze(0) for d in transformed_clips], dim=0)

        # Concatenate keypoints along the batch dimension??:
        batched_kp = torch.cat([d.unsqueeze(0) for d in unbatched_keypoints], dim=0)

        return dict(
            clips=batched_clips,  # (B, C, T, H, W), 
            keypoints=batched_kp,
            labels=torch.tensor(unbatched_labels),  # (K,)
            paths=paths,  # no need to make it a tensor
        )
