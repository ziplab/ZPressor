from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):

    view_rotation = extrinsics[:3, :3].inverse()
    # Apply the rotation to the means (Gaussian positions).
    means = einsum(view_rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = view_rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC band
    harmonics_view_invariant = harmonics[..., 0]

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    

def save_gaussian_ply(gaussians, visualization_dump, extrinsics, save_path, v, h, w):

    # Transform means into camera space.
    means = rearrange(
        gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
    )

    # Create a mask to filter the Gaussians. throw away Gaussians at the
    # borders, since they're generally of lower quality.
    mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
    GAUSSIAN_TRIM = 4
    mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

    def trim(element):
        element = rearrange(
            element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w
        )
        return element[mask][None]

    # convert the rotations from camera space to world space as required
    cam_rotations = trim(visualization_dump["rotations"])[0]
    c2w_mat = repeat(
        extrinsics[0, :, :3, :3],
        "v a b -> h w spp v a b",
        h=h,
        w=w,
        spp=1,
    )
    c2w_mat = c2w_mat[mask]  # apply trim

    cam_rotations_np = R.from_quat(
        cam_rotations.detach().cpu().numpy()
    ).as_matrix()
    world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
    world_rotations = R.from_matrix(world_mat).as_quat()
    world_rotations = torch.from_numpy(world_rotations).to(
        visualization_dump["scales"]
    )

    export_ply(
        extrinsics[0, 0],
        trim(gaussians.means)[0],
        trim(visualization_dump["scales"])[0],
        world_rotations,
        trim(gaussians.harmonics)[0],
        trim(gaussians.opacities)[0],
        save_path,
    )


