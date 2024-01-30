import torch
import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import Sequence
from PIL import Image


class ISBISegment(Dataset):
    def __init__(self, root: str, split: str = "train") -> None:
        self.root = root
        self.img_paths = sorted(
            [
                p
                for p in glob.glob(os.path.join(root, "images/**"), recursive=True)
                if os.path.isfile(p)
            ]
        )
        self.seg_paths = sorted(
            [
                p
                for p in glob.glob(os.path.join(root, "labels/**"), recursive=True)
                if os.path.isfile(p)
            ]
        )
        self.transform = ToTensor()

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index) -> Sequence[torch.Tensor]:
        img = Image.open(self.img_paths[index]).convert("L")
        img = self.transform(img)
        seg = Image.open(self.seg_paths[index]).convert("L")
        seg = self.transform(seg)
        seg[seg < 0.5] = 0
        seg[seg >= 0.5] = 1
        # TODO: load weight maps

        return img, seg
