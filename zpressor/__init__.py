"""
ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS

This package provides the ZPressor module for compressing multi-view inputs
in feed-forward 3D Gaussian Splatting applications.
"""

__version__ = "0.1.0"
__author__ = "Weijie Wang, Donny Y. Chen, Zeyu Zhang, Duochao Shi, Akide Liu, Bohan Zhuang"
__email__ = "wangweijie@zju.edu.cn"

from .zpressor import ZPressor
from .attention import CrossAttention, Attention
from .utils import center_filter

__all__ = [
    "ZPressor",
    "CrossAttention",
    "Attention",
    "center_filter",
]