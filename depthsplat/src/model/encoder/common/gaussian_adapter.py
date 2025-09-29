from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        assert input_images is not None

        scales = torch.clamp(torch.exp(scales - 4), max=self.cfg.gaussian_scale_max)

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
