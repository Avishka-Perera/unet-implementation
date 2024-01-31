from typing import Dict
from torch import nn
import torch


class SegmentCrossEntropy:
    def __init__(self, device: int = None) -> None:
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

    def __call__(
        self, info: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        lbl = batch["seg"]
        logits = info["seg"]

        if self.device is not None:
            lbl = lbl.to(self.device)
            logits = logits.to(self.device)

        lbl = lbl.squeeze()
        perm_dims = list(range(len(logits.shape)))
        c_dim = perm_dims.pop(-3)
        perm_dims.append(c_dim)
        logits = logits.permute(*perm_dims)

        lbl = lbl.reshape(-1)
        logits = logits.reshape(-1, 2)

        output = self.loss_fn(logits, lbl)

        return {"tot": output}
