import torch
import torch.nn as nn
import torch.nn.functional as F

from zpressor import ZPressor
from zpressor.utils import center_filter
from .backbone import CNNEncoder
from .vit_fpn import ViTFeaturePyramid
from .mv_transformer import (
    MultiViewFeatureTransformer,
    batch_features_camera_parameters,
)
from .matching import warp_with_pose_depth_candidates
from .utils import mv_feature_add_position
from .dpt_head import DPTHead
from .ldm_unet.unet import UNetModel, AttentionBlock
from einops import rearrange
import timm
from ....global_cfg import get_cfg


class MultiViewUniMatch(nn.Module):
    def __init__(
        self,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        num_zpressor_layers=6,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        grid_sample_disable_cudnn=False,
        no_cross_attn=False,
        use_cost_volume=False,
        use_cluster=True,
        apply_cluster_steps=0,
        use_clstoken=False,
        use_avg_token=False,
        no_self_attn=False,
        **kwargs,
    ):
        super(MultiViewUniMatch, self).__init__()

        self.use_cost_volume = use_cost_volume
        self.use_cluster = use_cluster
        self.apply_cluster_steps = apply_cluster_steps

        # CNN
        self.feature_channels = feature_channels
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        # monocular backbones: final
        self.vit_type = vit_type

        # cost volume
        self.num_depth_candidates = num_depth_candidates

        # upsampler
        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        #TODO CNN ---> dinov2
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        # zpressor
        self.use_clstoken = use_clstoken
        self.use_avg_token = use_avg_token
        if self.use_cluster:
            self.zipmatch = ZPressor(
                embed_dim=feature_channels,
                num_heads=num_head,
                num_layers=num_zpressor_layers,
                no_self_attn=no_self_attn
            )
            # self.fusion_layer = nn.Conv2d(feature_channels * 2, feature_channels, 1)


        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )

        # monodepth
        encoder = vit_type  # can also be 'vitb' or 'vitl'
        self.pretrained = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{:}14".format(encoder)
        )
        print("###########pretrained model#############")
        print(self.pretrained)
        
        if self.use_clstoken:
            resize_h, resize_w = get_cfg().dataset.image_shape[0] // 14 * 14, get_cfg().dataset.image_shape[1] // 14 * 14
            self.vit_model_name = "vit_base_patch14_dinov2.lvd142m"
            self.dinov2 = timm.create_model(
                self.vit_model_name,
                pretrained=True,
                img_size=(resize_h, resize_w),
            )
        # print("###########load model#############")
        # print(self.model)

        del self.pretrained.mask_token  # unused

        # UNet regressor
        self.regressor = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head = nn.ModuleList()

        i = 0
        curr_depth_candidates = num_depth_candidates // (4**i)
        cnn_feature_channels = 128 - (32 * i)
        mv_transformer_feature_channels = 128 // (2**i)

        mono_feature_channels = vit_feature_channel // (2**i)

        # concat(cost volume, cnn feature, mv feature, mono feature)
        if self.use_cost_volume:
            in_channels = (
                curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )
        else:
            in_channels = (
                # curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

        # unet channels
        channels = unet_channels // (2**i)

        # unet channel mult & unet_attn_resolutions
        if i > 0:
            unet_channel_mult = unet_channel_mult + [1]
            unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

        # unet
        modules = [
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        ]

        modules.append(
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=unet_num_res_blocks,
                attention_resolutions=unet_attn_resolutions,
                channel_mult=unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=False,
                num_frames=2,
                use_cross_view_self_attn=True,
            )
        )

        modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

        self.regressor.append(nn.Sequential(*modules))

        # regressor residual
        self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))

        # depth head
        self.depth_head.append(
            nn.Sequential(
                nn.Conv2d(
                    channels, channels * 2, 3, 1, 1, padding_mode="replicate"
                ),
                nn.GELU(),
                nn.Conv2d(
                    channels * 2,
                    curr_depth_candidates,
                    3,
                    1,
                    1,
                    padding_mode="replicate",
                ),
            )
        )

        # upsampler
        # concat(lowres_depth, cnn feature, mv feature, mono feature)
        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.upsampler = DPTHead(
            **model_configs[vit_type],
            downsample_factor=upsample_factor,
        )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        # images: [B, V, C, H, W]
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        # list of [BV, C, H, W], resolution from high to low
        features = self.backbone(concat)
        # reverse: resolution from low to high
        features = features[::-1]

        return features

    def forward(
        self,
        images,
        global_step,
        attn_splits_list=None,
        intrinsics=None,
        min_depth=1.0 / 0.5,  # inverse depth range
        max_depth=1.0 / 100,
        num_depth_candidates=128,
        extrinsics=None,
        is_nn_matrix=None,
        costvolume_nearest_n_views=0,
        cluster_num=8,
        **kwargs,
    ):
        self.cluster_num = cluster_num

        results_dict = {}
        depth_preds = []
        match_probs = []

        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14

        if not self.use_cluster:
            self.cluster_num = v

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.regressor, num_views=self.cluster_num)

        # NOTE: in this codebase, intrinsics are normalized by image width and height
        # in unimatch's codebase: https://github.com/autonomousvision/unimatch, no normalization
        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0] *= ori_w
        intrinsics[:, :, 1] *= ori_h

        # list of features, resolution low to high
        # list of [BV, C, H, W]
        features_list_cnn = self.extract_feature(images)
        features_list_cnn_all_scales = features_list_cnn
        # features_list_cnn = features_list_cnn[0]

        # mv transformer features
        # add position to features
        attn_splits = attn_splits_list[0]

        # [BV, C, H, W]
        features_cnn_pos = mv_feature_add_position(
            features_list_cnn[0], attn_splits, self.feature_channels
        )

        features_cnn_pos = rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v)

        if self.use_clstoken:
            # mono feature cls_token
            concat = rearrange(images, "b v c h w -> (b v) c h w")
            concat = F.interpolate(
                concat, (resize_h, resize_w), mode="bilinear", align_corners=True
            )
            dino_feat = self.dinov2.forward_features(concat)
            cls_token = dino_feat[:, 0]
            cls_token = rearrange(cls_token, "(b v) c -> b v c", b=b, v=v)
        elif self.use_avg_token:
            # Average the h w dimensions of features_cnn_pos
            cls_token = features_cnn_pos.mean(dim=3).mean(dim=3)
        else:
            cls_token = None
        

        # [B, V, C, H, W]
        features_list_cnn_all_scales = [rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v) for features in features_list_cnn_all_scales]
        features_list_cnn = [rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v) for features in features_list_cnn]
        if self.use_cluster:
            matched_feat, center_views = self.zipmatch(features_cnn_pos, extrinsics=extrinsics, cls_token=cls_token, cluster_num=self.cluster_num)
            
            results_dict.update({"center_views": center_views})
            if global_step >= self.apply_cluster_steps:
                clustered_features = matched_feat
            else:
                clustered_features = center_filter(features_cnn_pos, center_views)
        else:
            center_views = None
            clustered_features = center_filter(features_cnn_pos, center_views)

        # center images and extrinsics, intrinsics
        images = center_filter(images, center_views)
        extrinsics = center_filter(extrinsics, center_views)
        intrinsics = center_filter(intrinsics, center_views)
        max_depth = center_filter(max_depth, center_views)
        min_depth = center_filter(min_depth, center_views)

        # center features
        features_list_cnn_all_scales = center_filter(
            features_list_cnn_all_scales, center_views
        )
        features_list_cnn = center_filter(features_list_cnn, center_views)
        if is_nn_matrix:
            with torch.no_grad():
                xyzs = extrinsics[:, :, :3, -1].detach()
                nn_matrix = torch.cdist(xyzs, xyzs, p=2)
                nn_matrix = torch.argsort(nn_matrix)
                nn_matrix = nn_matrix[:, :, :costvolume_nearest_n_views]
        else:
            nn_matrix = None

        # [B, N, C, H, W] -> [BN, C, H, W]
        features_list_cnn_all_scales = [rearrange(features, "b n c h w -> (b n) c h w") for features in features_list_cnn_all_scales]
        features_list_cnn = [rearrange(features, "b n c h w -> (b n) c h w") for features in features_list_cnn]

        # store cnn features
        results_dict.update({"features_cnn_all_scales": features_list_cnn_all_scales})
        results_dict.update({"features_cnn": features_list_cnn})


        # list of [B, C, H, W]
        features_list = list(
            torch.unbind(
                clustered_features, dim=1
            )
        )
        features_list_mv = self.transformer(
            features_list,
            attn_num_splits=attn_splits,
            nn_matrix=nn_matrix,
        )

        features_mv = rearrange(
            torch.stack(features_list_mv, dim=1), "b v c h w -> (b v) c h w"
        )  # [BN, C, H, W]

        features_list_mv = [features_mv]

        results_dict.update({"features_mv": features_list_mv})

        # max_depth, min_depth: [B, N] -> [BN]
        max_depth = max_depth.view(-1)
        min_depth = min_depth.view(-1)

        # mono feature
        concat = rearrange(images, "b n c h w -> (b n) c h w")
        concat = F.interpolate(
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        )

        # get intermediate features
        intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }

        mono_intermediate_features = list(
            self.pretrained.get_intermediate_layers(
                concat, intermediate_layer_idx[self.vit_type], return_class_token=False
            )
        )
        # _, mono_intermediate_features = self.pretrained.forward_intermediates(
        #     concat, intermediate_layer_idx[self.vit_type]
        # )

        for i in range(len(mono_intermediate_features)):
            curr_features = (
                mono_intermediate_features[i]
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # resize to 1/8 resolution
            curr_features = F.interpolate(
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            mono_intermediate_features[i] = curr_features

        results_dict.update({"features_mono_intermediate": mono_intermediate_features})

        # last mono feature
        mono_features = mono_intermediate_features[-1]

        if self.lowest_feature_resolution == 4:
            mono_features = F.interpolate(
                mono_features, scale_factor=2, mode="bilinear", align_corners=True
            )

        features_list_mono = [mono_features]

        results_dict.update({"features_mono": features_list_mono})

        depth = None

        scale_idx = 0
        h, w = features_list_cnn[scale_idx].shape[-2:]
        downsample_factor = self.upsample_factor * (
            2 ** (-scale_idx)
        )

        if self.use_cost_volume:
            # scale intrinsics
            intrinsics_curr = intrinsics.clone()  # [B, N, 3, 3]
            intrinsics_curr[:, :, :2] = intrinsics_curr[:, :, :2] / downsample_factor

            # build cost volume
            features_mv = features_list_mv[scale_idx]  # [BN, C, H, W]

            # list of [B, C, H, W]
            features_mv_curr = list(
                torch.unbind(
                    rearrange(features_mv, "(b n) c h w -> b n c h w", b=b, n=self.cluster_num), dim=1
                )
            )

            intrinsics_curr = list(
                torch.unbind(intrinsics_curr, dim=1)
            )  # list of [B, 3, 3]
            extrinsics_curr = list(torch.unbind(extrinsics, dim=1))  # list of [B, 4, 4]

            # ref: [BV, C, H, W], [BV, 3, 3], [BV, 4, 4]
            # tgt: [BV, V-1, C, H, W], [BV, V-1, 3, 3], [BV, V-1, 4, 4]
            (
                ref_features,
                ref_intrinsics,
                ref_extrinsics,
                tgt_features,
                tgt_intrinsics,
                tgt_extrinsics,
            ) = batch_features_camera_parameters(
                features_mv_curr,
                intrinsics_curr,
                extrinsics_curr,
                nn_matrix=nn_matrix,
            )

            b_new, _, c, h, w = tgt_features.size()

            # relative pose
            # extrinsics: c2w
            pose_curr = torch.matmul(
                tgt_extrinsics.inverse(), ref_extrinsics.unsqueeze(1)
            )  # [BV, N-1, 4, 4]

        if scale_idx > 0:
            # 2x upsample depth
            assert depth is not None
            depth = F.interpolate(
                depth, scale_factor=2, mode="bilinear", align_corners=True
            ).detach()

        num_depth_candidates = self.num_depth_candidates // (4**scale_idx)

        # generate depth candidates
        if scale_idx == 0:
            # min_depth, max_depth: [BV]
            depth_interval = (max_depth - min_depth) / (
                self.num_depth_candidates - 1
            )  # [BN]

            linear_space = (
                torch.linspace(0, 1, num_depth_candidates)
                .type_as(features_list_cnn[0])
                .view(1, num_depth_candidates, 1, 1)
            )  # [1, D, 1, 1]

            depth_candidates = min_depth.view(-1, 1, 1, 1) + linear_space * (
                max_depth - min_depth
            ).view(
                -1, 1, 1, 1
            )  # [BN, D, 1, 1]
        else:
            # half interval each scale
            depth_interval = (
                (max_depth - min_depth)
                / (self.num_depth_candidates - 1)
                / (2**scale_idx)
            )  # [BN]
            # [BN, 1, 1, 1]
            depth_interval = depth_interval.view(-1, 1, 1, 1)

            # [BN, 1, H, W]
            depth_range_min = (
                depth - depth_interval * (num_depth_candidates // 2)
            ).clamp(min=min_depth.view(-1, 1, 1, 1))
            depth_range_max = (
                depth + depth_interval * (num_depth_candidates // 2 - 1)
            ).clamp(max=max_depth.view(-1, 1, 1, 1))

            linear_space = (
                torch.linspace(0, 1, num_depth_candidates)
                .type_as(features_list_cnn[0])
                .view(1, num_depth_candidates, 1, 1)
            )  # [1, D, 1, 1]
            depth_candidates = depth_range_min + linear_space * (
                depth_range_max - depth_range_min
            )  # [BN, D, H, W]

        if self.use_cost_volume:
            if scale_idx == 0:
                # [BN*(N-1), D, H, W]
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, h, w)
                    .view(-1, num_depth_candidates, h, w)
                )
            else:
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, 1, 1)
                    .view(-1, num_depth_candidates, h, w)
                )

            intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(
                -1, 3, 3
            )  # [BN, 3, 3]
            intrinsics_input = intrinsics_input.unsqueeze(1).repeat(
                1, tgt_features.size(1), 1, 1
            )  # [BN, V-1, 3, 3]

            warped_tgt_features = warp_with_pose_depth_candidates(
                rearrange(tgt_features, "b n ... -> (b n) ..."),
                rearrange(intrinsics_input, "b n ... -> (b n) ..."),
                rearrange(pose_curr, "b n ... -> (b n) ..."),
                1.0 / depth_candidates_curr,  # convert inverse depth to depth
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            )  # [BN*(N-1), C, D, H, W]

            # ref: [BN, C, H, W]
            # warped: [BN*(N-1), C, D, H, W] -> [BN, N-1, C, D, H, W]
            warped_tgt_features = rearrange(
                warped_tgt_features,
                "(b n) ... -> b n ...",
                b=b_new,
                n=tgt_features.size(1),
            )

        # regressor
        features_cnn = features_list_cnn[scale_idx]  # [BN, C, H, W]

        features_mono = features_list_mono[scale_idx]  # [BN, C, H, W]

        # [BN, N-1, D, H, W] -> [BN, D, H, W]
        # average cross other views
        if self.use_cost_volume:
            cost_volume = (
                (ref_features.unsqueeze(-3).unsqueeze(1) * warped_tgt_features).sum(2)
                / (c**0.5)
            ).mean(1)
            concat = torch.cat(
                (cost_volume, features_cnn, features_mv, features_mono), dim=1
            )
        else:
            print(features_cnn.shape, features_mv.shape, features_mono.shape)
            concat = torch.cat(
                (features_cnn, features_mv, features_mono), dim=1
            )

        out = self.regressor[scale_idx](concat) + self.regressor_residual[
            scale_idx
        ](concat)

        # depth pred
        match_prob = F.softmax(
            self.depth_head[scale_idx](out), dim=1
        )  # [BN, D, H, W]
        match_probs.append(match_prob)

        if scale_idx == 0:
            # [BN, D, H, W]
            depth_candidates = depth_candidates.repeat(1, 1, h, w)
        depth = (match_prob * depth_candidates).sum(
            dim=1, keepdim=True
        )  # [BN, 1, H, W]

        # upsample to the original resolution for supervison at training time only
        if self.training:
            depth_bilinear = F.interpolate(
                depth,
                scale_factor=downsample_factor,
                mode="bilinear",
                align_corners=True,
            )
            depth_preds.append(depth_bilinear)

        # final output, learned upsampler
        residual_depth = self.upsampler(
            mono_intermediate_features,
            # resolution high to low
            cnn_features=features_list_cnn_all_scales[::-1],
            mv_features=(
                features_mv
            ),
            depth=depth,
        )

        depth_bilinear = F.interpolate(
            depth,
            scale_factor=self.upsample_factor,
            mode="bilinear",
            align_corners=True,
        )
        depth = (depth_bilinear + residual_depth).clamp(
            min=min_depth.view(-1, 1, 1, 1), max=max_depth.view(-1, 1, 1, 1)
        )

        depth_preds.append(depth)

        # convert inverse depth to depth
        for i in range(len(depth_preds)):
            depth_pred = 1.0 / depth_preds[i].squeeze(1)  # [BN, H, W]
            depth_preds[i] = rearrange(
                depth_pred, "(b n) ... -> b n ...", b=b, n=self.cluster_num
            )  # [B, N, H, W]

        results_dict.update({"depth_preds": depth_preds})
        results_dict.update({"match_probs": match_probs})

        return results_dict
    
    def process_depths(self, b, v, depths, near, far):
        near = rearrange(near, "b v -> b v 1 1")
        far = rearrange(far, "b v -> b v 1 1")
        depths = rearrange(depths, "(b n) h w -> b n h w", b=b, n=self.cluster_num)
        
        eposilon = 1e-9

        min_vals = depths.amin(dim=(1, 2, 3), keepdim=True)
        max_vals = depths.amax(dim=(1, 2, 3), keepdim=True)
        
        # Normalize using the min/max per (h, w) map
        normalized_depths = (depths - min_vals) / (max_vals - min_vals + eposilon)
        normalized_depths = normalized_depths * (far - near) + near

        normalized_depths = rearrange(normalized_depths, "b n h w -> (b n) h w")

        return normalized_depths


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
        isinstance(module, nn.ModuleList)
        or isinstance(module, nn.Sequential)
        or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)
