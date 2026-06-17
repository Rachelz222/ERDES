import logging
import os
import random
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.io as io
from torch.utils.data import Dataset

from .utils import resize

log = logging.getLogger(__name__)


class VideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        size: Union[int, Tuple[int, int, int]],  # Desired (D, H, W)
        data_root: str = "",
        video_column: str = "path",
        label_column: str = "label",
    ):
        self.df = pd.read_csv(csv_path)
        self.video_paths = self.df[video_column].tolist()
        self.labels = self.df[label_column].tolist()
        self.resize_tf = resize(size)
        self.size = size
        self.data_root = data_root
        self._bad_indices = set()

    def __len__(self):
        return len(self.video_paths)

    def _load_video_safe(self, video_path: str) -> torch.Tensor:
        """Try to load video, falling back to a dummy frame on decode errors."""
        try:
            video, _, _ = io.read_video(video_path, pts_unit='sec')
            if video.numel() == 0:
                raise ValueError("Empty video")
            return video
        except Exception:
            # Some ultrasound clips have unusual pixel formats (e.g. yuv420p)
            # that torchvision cannot decode. Return a zero tensor as fallback.
            log.warning("Failed to decode %s, using zero tensor fallback", video_path)
            # Generate placeholder: 32 frames of zeros at default size
            D, H, W = self.size
            return torch.zeros(D, H, W, 3, dtype=torch.uint8)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        if self.data_root:
            video_path = os.path.join(self.data_root, video_path)
        label = self.labels[idx]

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        video = self._load_video_safe(video_path)

        # Convert to float tensor and permute to [C, D, H, W]
        video = video.float()
        video = video.permute(3, 0, 1, 2)  # [C, D, H, W]

        # If video has 3 channels, convert to grayscale by averaging
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)  # [1, D, H, W]

        # Apply transforms
        video = self.resize_tf(video)

        # Normalize
        video = video / 255.0

        label = torch.tensor(label, dtype=torch.float32)

        return video, label
