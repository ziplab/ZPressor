import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ResizeConvFeatureUpsampler(nn.Module):
    """
    https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self,
                 lowest_feature_resolution=8,
                 out_channels=128,
                 vit_type='vits',
                 no_mono_feature=False,
                 gaussian_downsample=None,
                 monodepth_backbone=False,
                 ):
        super(ResizeConvFeatureUpsampler, self).__init__()

        self.monodepth_backbone = monodepth_backbone

        self.upsampler = nn.ModuleList()

        vit_feature_channel_dict = {
            'vits': 384,
            'vitb': 768,
            'vitl': 1024
        }

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        if monodepth_backbone:
            vit_feature_channel = 384

        i = 0
        cnn_feature_channels = 128 - (32 * i)
        mv_transformer_feature_channels = 128 // (2 ** i)
        if no_mono_feature:
            mono_feature_channels = 0
        else:
            mono_feature_channels = vit_feature_channel // (2 ** i)

        in_channels = cnn_feature_channels + \
            mv_transformer_feature_channels + mono_feature_channels

        if monodepth_backbone:
            in_channels = 384

        curr_upsample_factor = lowest_feature_resolution // (2 ** i)

        num_upsample = int(math.log(curr_upsample_factor, 2))

        modules = []
        if num_upsample == 1:
            curr_in_channels = out_channels * 2
        else:
            curr_in_channels = out_channels * 2 * (num_upsample - 1)
        modules.append(nn.Conv2d(in_channels, curr_in_channels, 1))
        for i in range(num_upsample):
            modules.append(nn.Upsample(scale_factor=2, mode='nearest'))

            if i == num_upsample - 1:
                modules.append(nn.Conv2d(curr_in_channels,
                                            out_channels, 3, 1, 1, padding_mode='replicate'))
            else:
                modules.append(nn.Conv2d(curr_in_channels,
                                            curr_in_channels // 2, 3, 1, 1, padding_mode='replicate'))
                curr_in_channels = curr_in_channels // 2
                modules.append(nn.GELU())

        if gaussian_downsample is not None:
            if gaussian_downsample == 2:
                del modules[-3:]
            elif gaussian_downsample == 4:
                del modules[-6:]
            else:
                raise NotImplementedError

        self.upsampler.append(nn.Sequential(*modules))

    def forward(self, features_list_cnn, features_list_mv, features_list_mono=None):
        out = []

        i = 0
        if self.monodepth_backbone:
            concat = features_list_cnn[i]
        elif features_list_mono is None:
            concat = torch.cat(
            (features_list_cnn[i], features_list_mv[i]), dim=1)
        else:
            concat = torch.cat(
                (features_list_cnn[i], features_list_mv[i], features_list_mono[i]), dim=1)
        concat = self.upsampler[i](concat)

        out.append(concat)

        out = torch.cat(out, dim=1)

        return out


def _test():
    device = torch.device('cuda:0')
    
    model = ResizeConvFeatureUpsampler(lowest_feature_resolution=4,
                                       ).to(device)
    print(model)

    b, h, w = 2, 32, 64
    features_list_cnn = [torch.randn(b, 128, h, w).to(device)]
    features_list_mv = [torch.randn(b, 128, h, w).to(device)]
    features_list_mono = [torch.randn(b, 384, h, w).to(device)]

    # scale 2
    features_list_cnn.append(torch.randn(b, 96, h * 2, w * 2).to(device))
    features_list_mv.append(torch.randn(b, 64, h * 2, w * 2).to(device))
    features_list_mono.append(torch.randn(b, 192, h * 2, w * 2).to(device))

    out = model(features_list_cnn,
                features_list_mv, features_list_mono)

    print(out.shape)


if __name__ == '__main__':
    _test()
