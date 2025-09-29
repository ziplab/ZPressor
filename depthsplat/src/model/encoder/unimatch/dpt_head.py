import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        concat_cnn_features=True,
        concat_mv_features=True,
        cnn_feature_channels=[64, 96, 128],
        concat_features=True,
        downsample_factor=8,
        return_feature=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.concat_cnn_features = concat_cnn_features
        self.concat_mv_features = concat_mv_features
        self.concat_features = concat_features
        self.downsample_factor = downsample_factor
        self.return_feature = return_feature

        if self.concat_features:
            if self.downsample_factor == 4:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0]
                            + cnn_feature_channels[1]
                            + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2]
                            + out_channels[1]
                            + 128
                            + depth_channel,
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(out_channels[2], out_channels[2], 1),  # 1/8 mono
                    ]
                )
            else:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0] + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[1] + out_channels[1],
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2]
                            + out_channels[2]
                            + 128
                            + depth_channel,
                            out_channels[2],
                            1,
                        ),  # 1/8 concat(cnn, mono, mv, depth)
                    ]
                )
        else:
            if self.concat_cnn_features:
                self.cnn_projects = nn.ModuleList(
                    [
                        nn.Conv2d(cnn_feature_channels[i], out_channels[i], 1)
                        for i in range(len(cnn_feature_channels))
                    ]
                )

            if self.concat_mv_features:
                self.mv_projects = nn.Conv2d(128, out_channels[2], 1)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # not used
        del self.scratch.refinenet4.resConfUnit1

        head_features_1 = features
        head_features_2 = 16

        if not self.return_feature:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(
                    head_features_1,
                    head_features_1 // 2,
                    3,
                    1,
                    1,
                    padding_mode="replicate",
                ),
                nn.GELU(),
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.GELU(),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            )

            # init delta depth as zero
            nn.init.zeros_(self.scratch.output_conv[-1].weight)
            nn.init.zeros_(self.scratch.output_conv[-1].bias)

    def forward(
        self,
        out_features,
        downsample_factor=8,
        cnn_features=None,
        mv_features=None,
        depth=None,
    ):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        # 1/2, 1/4, 1/8, 1/16
        layer_1, layer_2, layer_3, layer_4 = out

        if self.concat_features:
            if not self.return_feature:
                assert depth is not None

            if self.downsample_factor == 4:
                # print("concat_features len", len(cnn_features))
                # print("concat_feature0 shape", cnn_features[0].shape)
                # print("concat_feature1 shape", cnn_features[1].shape)
                # print("layer_1 shape", layer_1.shape)
                concat1 = torch.cat((cnn_features[0], cnn_features[1], layer_1), dim=1)
            else:
                concat1 = torch.cat((cnn_features[0], layer_1), dim=1)
            layer_1 = self.concat_projects[0](concat1)  # 1/2

            if self.downsample_factor == 4:
                if self.return_feature:
                    concat2 = torch.cat((cnn_features[2], layer_2, mv_features), dim=1)
                else:
                    concat2 = torch.cat(
                        (cnn_features[2], layer_2, mv_features, depth), dim=1
                    )
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                concat3 = layer_3
                layer_3 = self.concat_projects[2](concat3)  # 1/8
            else:
                concat2 = torch.cat((cnn_features[1], layer_2), dim=1)
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                if self.return_feature:
                    concat3 = torch.cat((cnn_features[2], layer_3, mv_features), dim=1)
                else:
                    concat3 = torch.cat(
                        (cnn_features[2], layer_3, mv_features, depth), dim=1
                    )
                layer_3 = self.concat_projects[2](concat3)  # 1/8
        else:
            if self.concat_cnn_features:
                assert cnn_features is not None
                assert len(cnn_features) == 3  # 1/2, 1/4, 1/8
                cnn_features = [
                    self.cnn_projects[i](f) for i, f in enumerate(cnn_features)
                ]

                layer_1 = layer_1 + cnn_features[0]  # 1/2
                layer_2 = layer_2 + cnn_features[1]  # 1/4
                layer_3 = layer_3 + cnn_features[2]  # 1/8

            if self.concat_mv_features:
                # 1/8
                mv_features = self.mv_projects(mv_features)

                layer_3 = layer_3 + mv_features  # 1/8

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])  # 1/8
        path_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:]
        )  # 1/4
        path_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:]
        )  # 1/2
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)  # 1

        if self.return_feature:
            return path_1

        out = self.scratch.output_conv(path_1)

        return out


if __name__ == "__main__":
    device = torch.device("cuda")
    c = 384
    model = DPTHead(
        in_channels=c,
        concat_cnn_features=True,
        concat_mv_features=True,
    ).to(device)
    print(model)

    h, w = 16, 32

    x = torch.randn(2, c, h, w).to(device)

    out_features = [x] * 4

    cnn_features = [
        torch.randn(2, 64, h * 4, w * 4).to(device),
        torch.randn(2, 96, h * 2, w * 2).to(device),
        torch.randn(2, 128, h, w).to(device),
    ]

    mv_features = torch.randn(2, 128, h, w).to(device)

    out = model(out_features, h, w, cnn_features=cnn_features, mv_features=mv_features)

    print(out.shape)
