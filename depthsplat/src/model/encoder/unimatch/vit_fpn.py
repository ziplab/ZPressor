import torch
import torch.nn as nn
import torch.nn.functional as F


# Ref: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py#L363


class ViTFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        scale_factors,
    ):
        """
        Args:
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super(ViTFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        out_dim = dim = in_channels
        self.stages = nn.ModuleList()
        for idx, scale in enumerate(scale_factors):
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            if scale != 1.0:
                layers.extend(
                    [
                        nn.GELU(),
                        nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                    ]
                )
            layers = nn.Sequential(*layers)

            self.stages.append(layers)

    def forward(self, x):
        results = []

        for stage in self.stages:
            results.append(stage(x))

        return results


def _test():
    model = ViTFeaturePyramid(
        384,
        scale_factors=[1, 2, 4],
    ).cuda()
    print(model)

    x = torch.randn(2, 384, 64, 96).cuda()

    out = model(x)

    for x in out:
        print(x.shape)


if __name__ == "__main__":
    _test()
