import torch
import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize, Pad, InterpolationMode
from typing import Sequence
from PIL import Image
import numpy as np
from .util.data import get_9_pt_flow, warp_image


class ISBISegment(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        rsz_hw: Sequence[int] = [388, 388],
        pad_hw: Sequence[int] = [572, 572],
        do_aug: bool = True,
        warp_std: float = 10.0,
    ) -> None:
        self.root = root
        self.split = split
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
            ]
        )
        self.seg_trans = Compose(
            [
                Resize(rsz_hw, InterpolationMode.NEAREST),
                Pad(padding=pad_hw, padding_mode="reflect"),
            ]
        )
        self.to_tensor = ToTensor()

        self.subject_region = (
            *pad_hw[::-1],
            rsz_hw[1] + pad_hw[1],
            rsz_hw[0] + pad_hw[0],
        )
        self.do_aug = do_aug
        self.warp_std = warp_std

    def __len__(self) -> int:
        return len(self.img_paths)

    def _warp_pair(self, img1, img2):
        flow = get_9_pt_flow(img1.shape[:-1], std=self.warp_std)
        img1 = warp_image(img1, flow)
        img2 = warp_image(img2, flow)
        return img1, img2

    def _rott_pair(self, img1, img2):
        num_rots = np.random.randint(4)
        img1 = np.rot90(img1, k=num_rots, axes=(0, 1))
        img2 = np.rot90(img2, k=num_rots, axes=(0, 1))
        return img1, img2

    def _random_aug(self, img1, img2):
        img1, img2 = self._rott_pair(img1, img2)
        img1, img2 = self._warp_pair(img1, img2)
        return img1, img2

    def __getitem__(self, index) -> Sequence[torch.Tensor]:
        img = Image.open(self.img_paths[index]).convert("L")
        img = np.array(self.img_trans(img))[..., np.newaxis]
        seg = Image.open(self.seg_paths[index]).convert("L")
        seg = np.array(self.seg_trans(seg))[..., np.newaxis]
        if self.do_aug:
            img, seg = self._random_aug(img, seg)
        seg = (seg / 255).astype(np.float32)
        img = (img / 255).astype(np.float32)
        seg[seg < 0.5] = 0
        seg[seg >= 0.5] = 1

        img = self.to_tensor(img)
        seg = self.to_tensor(seg)
        seg = seg.long()

        # TODO: load weight maps

        return {"img": img, "seg": seg}
