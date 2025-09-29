from typing import Optional

from .encoder import Encoder
from .encoder_depthsplat import EncoderDepthSplat, EncoderDepthSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_depthsplat import EncoderVisualizerDepthSplat

ENCODERS = {
    "depthsplat": (EncoderDepthSplat, EncoderVisualizerDepthSplat),
}

EncoderCfg = EncoderDepthSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
