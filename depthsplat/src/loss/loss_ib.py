from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .loss import Loss
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians


@dataclass
class LossIBCfg:
    weight: float


@dataclass
class LossIBWrapper:
    ib: LossIBCfg


class LossIB(Loss[LossIBCfg, LossIBWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        latent_z=None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        if latent_z is None:
            return torch.tensor(0.0, device=prediction.color.device, requires_grad=True)

        b, n, c, h, w = latent_z.shape
        latent_flat = latent_z.view(b * n, c, h * w).permute(0, 2, 1)  # [BN, HW, C]

        mu = latent_flat.mean(dim=1)
        var = latent_flat.var(dim=1, unbiased=False)
        std = torch.sqrt(var + 1e-8)

        kl_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2))
        kl_loss = kl_loss.sum(1).mean()

        kl_loss = kl_loss.div(math.log(2))

        return self.cfg.weight * kl_loss