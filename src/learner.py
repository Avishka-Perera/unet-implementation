import torch
from typing import Any, Dict, Sequence
from torch import nn


class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=0,
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class Connector:
    def __init__(self) -> None:
        self.slices = {}

    def _get_reshape_slice(self, dst_shape, src_shape):
        *_, H, W = src_shape
        *_, h, w = dst_shape
        ver_pad = int((H - h) / 2)
        hor_pad = int((W - w) / 2)
        ver_slice = slice(ver_pad, ver_pad + h)
        hor_slice = slice(hor_pad, hor_pad + w)
        return ver_slice, hor_slice

    def __call__(self, ten1: torch.Tensor, ten2: torch.Tensor) -> torch.Tensor:
        key = f"{ten1.shape}:{ten2.shape}"
        if key in self.slices:
            ver_slice, hor_slice = self.slices[key]
        else:
            ver_slice, hor_slice = self._get_reshape_slice(ten1.shape, ten2.shape)
            self.slices[key] = (ver_slice, hor_slice)
        ten2_cropped = ten2[..., ver_slice, hor_slice]
        cat = torch.concat([ten1, ten2_cropped], -3)

        return cat


class Contractive(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net_fiv = nn.Sequential(Conv3x3(1, 64), Conv3x3(64, 64))
        self.net_fou = nn.Sequential(
            nn.MaxPool2d(2), Conv3x3(64, 128), Conv3x3(128, 128)
        )
        self.net_thr = nn.Sequential(
            nn.MaxPool2d(2), Conv3x3(128, 256), Conv3x3(256, 256)
        )
        self.net_two = nn.Sequential(
            nn.MaxPool2d(2), Conv3x3(256, 512), Conv3x3(512, 512)
        )
        self.net_one = nn.Sequential(
            nn.MaxPool2d(2), Conv3x3(512, 1024), Conv3x3(1024, 1024)
        )

    def forward(self, im: torch.Tensor) -> Dict[str, torch.Tensor]:
        f5 = self.net_fiv(im)
        f4 = self.net_fou(f5)
        f3 = self.net_thr(f4)
        f2 = self.net_two(f3)
        f1 = self.net_one(f2)
        return {"f5": f5, "f4": f4, "f3": f3, "f2": f2, "f1": f1}


class Expansive(nn.Module):
    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.n_out = n_out
        self.upc_one = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.net_one = nn.Sequential(Conv3x3(1024, 512), Conv3x3(512, 512))
        self.upc_two = nn.ConvTranspose2d(512, 256, 2, 2)
        self.net_two = nn.Sequential(Conv3x3(512, 256), Conv3x3(256, 256))
        self.upc_thr = nn.ConvTranspose2d(256, 128, 2, 2)
        self.net_thr = nn.Sequential(Conv3x3(256, 128), Conv3x3(128, 128))
        self.upc_fou = nn.ConvTranspose2d(128, 64, 2, 2)
        self.net_fou = nn.Sequential(Conv3x3(128, 64), Conv3x3(64, 64))
        self.project = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1,
            padding=0,
        )
        self.conn = Connector()

    def forward(self, feat_pyr: Dict[str, torch.Tensor]) -> torch.Tensor:
        actvs = self.upc_one(feat_pyr["f1"])
        actvs = self.conn(actvs, feat_pyr["f2"])
        actvs = self.net_one(actvs)

        actvs = self.upc_two(actvs)
        actvs = self.conn(actvs, feat_pyr["f3"])
        actvs = self.net_two(actvs)

        actvs = self.upc_thr(actvs)
        actvs = self.conn(actvs, feat_pyr["f4"])
        actvs = self.net_thr(actvs)

        actvs = self.upc_fou(actvs)
        actvs = self.conn(actvs, feat_pyr["f5"])
        actvs = self.net_fou(actvs)

        actvs = self.project(actvs)

        return actvs


class UNet(nn.Module):
    device_count = 1

    def __init__(self, n_out: int = 2) -> None:
        super().__init__()
        self.n_out = n_out
        self.cont_path = Contractive()
        self.expa_path = Expansive(n_out=n_out)
        self.initialize_weights()
        self.device = None

    def set_devices(self, devices: Sequence[int]) -> None:
        self.device = devices[0]
        self.to(devices[0])

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.device is not None:
            batch["img"] = batch["img"].to(self.device)
        img = batch["img"]

        feat_pyr = self.cont_path(img)
        seg = self.expa_path(feat_pyr)

        return {"logits": seg}
