import io
from pathlib import Path
from typing import Union
import skvideo.io
import cv2

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(
    images: list[FloatImage],
    path: Union[Path, str],
    fps: None | int = None
) -> None:
    """Save an image. Assumed to be in range 0-1.
    images: [(3, h, w), (3, h, w), ...]
    """

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    # Image.fromarray(prep_image(image)).save(path)
    frames = []
    for image in images:
        frames.append(prep_image(image))

    outputdict = {'-pix_fmt': 'yuv420p', '-crf': '23',
                  '-vf': f'setpts=1.*PTS'}
                  
    if fps is not None:
        outputdict.update({'-r': str(fps)})

    writer = skvideo.io.FFmpegWriter(path,
                                     outputdict=outputdict)
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()


def save_video_opencv(
    images: list[FloatImage],
    path: Union[Path, str],
    fps: int = 30,
    codec: str = 'mp4v'
) -> None:
    """Save video using OpenCV

    Args:
        images: List of images, range 0-1, format [(3, h, w), (3, h, w), ...]
        path: Output video path
        fps: Frame rate, default 30
        codec: Video codec, default 'mp4v'
    """
    # Create output directory
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Convert all images to numpy arrays
    frames = [prep_image(image) for image in images]

    # Get video dimensions
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        # Convert RGB to BGR (OpenCV uses BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release video writer
    out.release()

if __name__ == "__main__":
    # Write a function to test the save_video function
    images = [torch.randn(3, 256, 256), torch.randn(3, 256, 256), torch.randn(3, 256, 256)]
    save_video(images, "test.mp4")
