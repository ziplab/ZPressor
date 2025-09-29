from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    extra_info: dict[str, Tensor | None] | None = None
