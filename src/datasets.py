import torch
import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize, Pad
from typing import Sequence
from PIL import Image


class ISBISegment(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        rsz_hw: Sequence[int] = [388, 388],
        pad_hw: Sequence[int] = [572, 572],
    ) -> None:
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
        pad_hw = [int((p - r) / 2) for r, p in zip(rsz_hw, pad_hw)]
        self.img_trans = Compose(
            [
                Resize(rsz_hw),
                Pad(padding=pad_hw, padding_mode="reflect"),
                ToTensor(),
            ]
        )
        self.seg_trans = Compose(
            [
                Resize(rsz_hw),
                ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index) -> Sequence[torch.Tensor]:
        img = Image.open(self.img_paths[index]).convert("L")
        img = self.img_trans(img)
        seg = Image.open(self.seg_paths[index]).convert("L")
        seg = self.seg_trans(seg)
        seg[seg < 0.5] = 0
        seg[seg >= 0.5] = 1
        seg = seg.long()

        # TODO: load weight maps

        return {"img": img, "seg": seg}
