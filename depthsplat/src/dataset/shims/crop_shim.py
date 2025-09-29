import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import torch.nn.functional as F

from ..types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: None | Float[Tensor, "*#batch h w"],
) -> (
    tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
    | tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
        Float[Tensor, "*#batch h_out w_out"],  # updated depths
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    if depths is not None:
        depths = depths[..., :, row : row + h_out, col : col + w_out]
        return images, intrinsics, depths

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: None | Float[Tensor, "*#batch h w"],
) -> (
    tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
    | tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
        Float[Tensor, "*#batch h_out w_out"],  # updated depths
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    # reshape and crop depth as well when available
    if depths is not None:
        depths = F.interpolate(
            depths.unsqueeze(1),
            size=(h_scaled, w_scaled),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

    return center_crop(images, intrinsics, shape, depths=depths)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    images, intrinsics = rescale_and_crop(
        views["image"], views["intrinsics"], shape, depths=None
    )
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }
