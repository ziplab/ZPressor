from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_ib import LossIB, LossIBWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossIBWrapper: LossIB,
}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper | LossIBWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
