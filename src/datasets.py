import torch
import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize, Pad, InterpolationMode
from typing import Sequence
from PIL import Image
import numpy as np
from .util.data import get_9_pt_flow, warp_image


# Downloaded from https://github.com/hoangp/isbi-datasets/tree/master (ISBI EM) and http://celltrackingchallenge.net/2d-datasets/ (DIC-HeLa and PhC-U373)


class Base(Dataset):
    valid_splits = ["train", "val"]
    split_ratio = [0.8, 0.2]

    def __init__(
        self,
        root: str,
        split: str = "train",
        rsz_hw: Sequence[int] = [388, 388],
        pad_hw: Sequence[int] = [572, 572],
        do_aug: bool = True,
        trim_seg: bool = True,
        warp_std: float = 10.0,
        img_dirs=["images"],
        lbl_dirs=["labels"],
    ) -> None:
        assert split in self.valid_splits
        assert len(img_dirs) == len(lbl_dirs)

        self.root = root
        self.split = split
        img_paths = sorted(
            [
                p
                for dir in img_dirs
                for p in glob.glob(os.path.join(root, dir, "**"), recursive=True)
                if os.path.isfile(p)
            ]
        )
        seg_paths = sorted(
            [
                p
                for dir in lbl_dirs
                for p in glob.glob(os.path.join(root, dir, "**"), recursive=True)
                if os.path.isfile(p)
            ]
        )
        if split == "train":
            self.img_paths = img_paths[: int(len(img_paths) * self.split_ratio[0])]
            self.seg_paths = seg_paths[: int(len(seg_paths) * self.split_ratio[0])]
        else:
            self.img_paths = img_paths[int(len(img_paths) * self.split_ratio[0]) :]
            self.seg_paths = seg_paths[int(len(seg_paths) * self.split_ratio[0]) :]

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
        self.trim_seg = trim_seg
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


class ISBISegment(Base):
    def __init__(
        self,
        root: str,
        split: str = "train",
        rsz_hw: Sequence[int] = [388, 388],
        pad_hw: Sequence[int] = [572, 572],
        do_aug: bool = True,
        trim_seg: bool = True,
        warp_std: float = 10,
    ) -> None:
        super().__init__(
            root,
            split,
            rsz_hw,
            pad_hw,
            do_aug,
            trim_seg,
            warp_std,
            img_dirs=["images"],
            lbl_dirs=["labels"],
        )

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
        if self.trim_seg:
            seg = seg[
                self.subject_region[1] : self.subject_region[3],
                self.subject_region[0] : self.subject_region[2],
            ]

        img = self.to_tensor(img)
        seg = self.to_tensor(seg)
        seg = seg.long()

        # TODO: load weight maps

        return {"img": img, "seg": seg}


class ISBICellTrack(Base):
    def __init__(
        self,
        root: str,
        split: str = "train",
        rsz_hw: Sequence[int] = [388, 388],
        pad_hw: Sequence[int] = [572, 572],
        do_aug: bool = True,
        trim_seg: bool = True,
        warp_std: float = 10,
    ) -> None:
        super().__init__(
            root,
            split,
            rsz_hw,
            pad_hw,
            do_aug,
            trim_seg,
            warp_std,
            img_dirs=["01", "02"],
            lbl_dirs=["01_ST/SEG", "02_ST/SEG"],
        )

    def __getitem__(self, index) -> Sequence[torch.Tensor]:
        img = Image.open(self.img_paths[index]).convert("L")
        img = np.array(self.img_trans(img))[..., np.newaxis]
        seg_tmp = Image.open(self.seg_paths[index])
        seg_tmp = np.array(self.seg_trans(seg_tmp))
        seg = np.ones(seg_tmp.shape).astype(np.float32)
        seg[seg_tmp == 0] = 0
        seg = seg[..., np.newaxis]
        if self.do_aug:
            img, seg = self._random_aug(img, seg)
        img = (img / 255).astype(np.float32)
        seg[seg < 0.5] = 0
        seg[seg >= 0.5] = 1
        if self.trim_seg:
            seg = seg[
                self.subject_region[1] : self.subject_region[3],
                self.subject_region[0] : self.subject_region[2],
            ]

        img = self.to_tensor(img)
        seg = self.to_tensor(seg)
        seg = seg.long()

        # TODO: load weight maps

        return {"img": img, "seg": seg}
