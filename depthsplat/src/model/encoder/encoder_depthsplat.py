from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
import random
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch, set_num_views
from .unimatch.ldm_unet.unet import UNetModel
from .unimatch.feature_upsampler import ResizeConvFeatureUpsampler

from zpressor.utils import center_filter


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int
    no_mapping: bool


@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool
    no_cross_attn: bool
    use_cost_volume: bool
    use_cluster: bool
    cluster_num: int
    min_cluster_num: int
    max_cluster_num: int
    num_zpressor_layers: int
    cluster_group: List[int]
    apply_cluster_steps: int
    use_clstoken: bool
    use_avg_token: bool
    no_self_attn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    costvolume_nearest_n_views: Optional[int] = None
    multiview_trans_nearest_n_views: Optional[int] = None


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
            no_cross_attn=cfg.no_cross_attn,
            use_cost_volume=cfg.use_cost_volume,
            use_cluster=cfg.use_cluster,
            apply_cluster_steps=cfg.apply_cluster_steps,
            use_clstoken=cfg.use_clstoken,
            use_avg_token=cfg.use_avg_token,
            no_self_attn=cfg.no_self_attn,
            num_zpressor_layers=cfg.num_zpressor_layers
        )

        if self.cfg.train_depth_only:
            return

        # upsample to the original resolution
        self.feature_upsampler = ResizeConvFeatureUpsampler(lowest_feature_resolution=cfg.lowest_feature_resolution,
                                                            out_channels=self.cfg.feature_upsampler_channels,
                                                            vit_type=self.cfg.monodepth_vit_type,
                                                            )
        feature_upsampler_channels = self.cfg.feature_upsampler_channels
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # unet
        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        modules = [
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        ]

        if self.cfg.color_large_unet or self.cfg.gaussian_regressor_channels == 16:
            unet_channel_mult = [1, 2, 4, 4, 4]
        else:
            unet_channel_mult = [1, 1, 1, 1, 1]
        unet_attn_resolutions = [16]

        modules.append(
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=unet_attn_resolutions,
                channel_mult=unet_channel_mult,
                num_head_channels=32 if self.cfg.gaussian_regressor_channels >= 32 else 16,
                dims=2,
                postnorm=False,
                num_frames=2,
                use_cross_view_self_attn=True,
            )
        )

        modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2

        # predict opacity
        num_gaussian_parameters += 1

        # concat(img, features, unet_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1

        if self.cfg.feature_upsampler_channels != 128:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                            3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                            num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )
        else:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(
                    in_channels, num_gaussian_parameters * 2, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters * 2,
                            num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):

        if self.cfg.min_cluster_num != 0 and self.cfg.max_cluster_num != 0:
            self.cfg.cluster_num = random.randint(self.cfg.min_cluster_num, self.cfg.max_cluster_num)
            # print("cluster num: ", self.cfg.cluster_num)
            
        # If cluster_group is not empty, use cluster_group and randomly select a cluster_num from it
        if self.cfg.cluster_group != []:
            self.cfg.cluster_num = random.choice(self.cfg.cluster_group)

        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        original_b = b  # Save original batch size for latent_z reshaping

        if not self.cfg.use_cluster:
            self.cfg.cluster_num = v

        # if (
        #     self.cfg.costvolume_nearest_n_views is not None
        #     or self.cfg.multiview_trans_nearest_n_views is not None
        # ):
        #     assert self.cfg.costvolume_nearest_n_views is not None
        #     with torch.no_grad():
        #         xyzs = context["extrinsics"][:, :, :3, -1].detach()
        #         cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
        #         cameras_dist_index = torch.argsort(cameras_dist_matrix)

        #         cameras_dist_index = cameras_dist_index[:,
        #                                                 :, :self.cfg.costvolume_nearest_n_views]
        # else:
        #     cameras_dist_index = None
        is_nn_matrix = self.cfg.costvolume_nearest_n_views is not None or self.cfg.multiview_trans_nearest_n_views is not None
        if is_nn_matrix:
            assert self.cfg.costvolume_nearest_n_views is not None

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            global_step,
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            is_nn_matrix=is_nn_matrix,
            costvolume_nearest_n_views=self.cfg.costvolume_nearest_n_views,
            cluster_num=self.cfg.cluster_num
        )

        if self.cfg.use_cluster:
            # clustered center views
            center_views = results_dict["center_views"]

            # process context to center views
            images = center_filter(context["image"], center_views)
            extrinsics = center_filter(context["extrinsics"], center_views)
            intrinsics = center_filter(context["intrinsics"], center_views)
        else:
            center_views = None
            images = context["image"]
            extrinsics = context["extrinsics"]
            intrinsics = context["intrinsics"]

        # list of [B, N, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, N, H, W]
        depth = depth_preds[-1]

        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b n h w -> b n (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, N, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b n h w -> b n (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b n (h w) srf s -> b n h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, N, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.gaussian_regressor, self.cfg.cluster_num)

        # features [BN, C, H, W]
        features = self.feature_upsampler(results_dict["features_cnn"],
                                            results_dict["features_mv"],
                                            results_dict["features_mono"],
                                            )

        # match prob from softmax
        # [BN, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BN, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        concat = torch.cat((
            rearrange(images, "b n c h w -> (b n) c h w"),
            rearrange(depth, "b n h w -> (b n) () h w"),
            match_prob,
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(images,
                            "b n c h w -> (b n) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)

        gaussians = self.gaussian_head(out)  # [BN, C, H, W]

        gaussians = rearrange(gaussians, "(b n) c h w -> b n c h w", b=b, n=self.cfg.cluster_num)

        depths = rearrange(depth, "b n h w -> b n (h w) () ()")

        # [B, N, H*W, 1, 1]
        densities = rearrange(
            match_prob, "(b n) c h w -> b n (c h w) () ()", b=b, n=self.cfg.cluster_num)
        # [B, N, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b n c h w -> b n (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)
            
            intermediate_depths = rearrange(
                intermediate_depths, "b n h w -> b n (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths

        # [B, N, H*W, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = images

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_extrinsics = torch.cat(
                [extrinsics] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [intrinsics] * len(depth_preds), dim=0)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b n i j -> b n () () () i j"),
                rearrange(context_intrinsics, "b n i j -> b n () () () i j"),
                rearrange(xy_ray, "b n r srf xy -> b n r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b n r srf c -> b n r srf () c",
                ),
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
            )

        else:
            gaussians = self.gaussian_adapter.forward(
                rearrange(extrinsics,
                          "b n i j -> b n () () () i j"),
                rearrange(intrinsics,
                          "b n i j -> b n () () () i j"),
                rearrange(xy_ray, "b n r srf xy -> b n r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b n r srf c -> b n r srf () c",
                ),
                (h, w),
                input_images=sh_input_images if self.cfg.init_sh_input_img else None,
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b n (h w) srf s -> b n h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b n r srf spp xyz -> b (n r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b n r srf spp xyzw -> b (n r srf spp) xyzw"
            )

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b n r srf spp xyz -> b (n r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b n r srf spp i j -> b (n r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b n r srf spp c d_sh -> b (n r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b n r srf spp -> b (n r srf spp)",
            ),
            extra_info={
                "center_views": center_views,
            },
        )

        ret = {}

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b n (h w) srf s -> b n h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, N, H, W]
            ret["depths"] = depths
        
        ret["latent_z"] = rearrange(features, "(b n) c h w -> b n c h w", b=original_b, n=self.cfg.cluster_num)
        
        ret["gaussians"] = gaussians

        return ret

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None
