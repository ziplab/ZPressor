# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9591))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
from pathlib import Path
import warnings
import copy

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from pytorch_lightning.plugins.environments import LightningEnvironment


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.resume_ckpt import find_latest_ckpt
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    if cfg_dict["mode"] == "train" and cfg_dict["train"]["eval_model_every_n_val"] > 0:
        eval_cfg_dict = copy.deepcopy(cfg_dict)
        dataset_dir = str(cfg_dict["dataset"]["roots"]).lower()
        if "re10k" in dataset_dir:
            num_context_views = cfg_dict["dataset"]["view_sampler"]["num_context_views"]
            num_target_views = cfg_dict["dataset"]["view_sampler"]["num_target_views"]
            max_distance_between_context_views = cfg_dict["dataset"]["view_sampler"]["max_distance_between_context_views"]
            eval_path = "assets/re10k_evaluation/re10k_ctx_" + str(num_context_views) + "v_tgt_" + str(num_target_views) + "v_n" + str(max_distance_between_context_views) + ".json"
        if "acid" in dataset_dir:
            num_context_views = cfg_dict["dataset"]["view_sampler"]["num_context_views"]
            num_target_views = cfg_dict["dataset"]["view_sampler"]["num_target_views"]
            max_distance_between_context_views = cfg_dict["dataset"]["view_sampler"]["max_distance_between_context_views"]
            eval_path = "assets/acid_evaluation/acid_ctx_" + str(num_context_views) + "v_tgt_" + str(num_target_views) + "v_n" + str(max_distance_between_context_views) + ".json"
        elif "dl3dv" in dataset_dir:
            num_context_views = cfg_dict["dataset"]["view_sampler"]["num_context_views"]
            num_target_views = cfg_dict["dataset"]["view_sampler"]["num_target_views"]
            max_distance_between_context_views = cfg_dict["dataset"]["view_sampler"]["max_distance_between_context_views"]
            eval_path = "assets/dl3dv_evaluation/dl3dv_ctx_" + str(num_context_views) + "v_tgt_" + str(num_target_views) + "v_n" + str(max_distance_between_context_views) + ".json"
        if not os.path.exists(eval_path):
            print(f"Fail to load eval index path: {eval_path}")
            raise Exception("Fail to load eval index path")
        eval_cfg_dict["dataset"]["view_sampler"] = {
            "name": "evaluation",
            "index_path": eval_path,
            "num_context_views": cfg_dict["dataset"]["view_sampler"]["num_context_views"],
        }
        eval_cfg = load_typed_root_config(eval_cfg_dict)
    else:
        eval_cfg = None

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=os.path.basename(cfg_dict.output_dir),
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    if cfg.checkpointing.resume:
        if not os.path.exists(output_dir / 'checkpoints'):
            checkpoint_path = None
        else:
            checkpoint_path = find_latest_ckpt(output_dir / 'checkpoints')
            print(f'resume from {checkpoint_path}')
    else:
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=torch.cuda.device_count(),
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        num_nodes=cfg.trainer.num_nodes,
        plugins=LightningEnvironment() if cfg.use_plugins else None,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        eval_data_cfg=(
            None if eval_cfg is None else eval_cfg.dataset
        ),
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        print("train:", len(data_module.train_dataloader()))
        print("val:", len(data_module.val_dataloader()))
        print("test:", len(data_module.test_dataloader()))

    strict_load = not cfg.checkpointing.no_strict_load

    if cfg.mode == "train":
        # only load monodepth
        if cfg.checkpointing.pretrained_monodepth is not None:
            strict_load = False
            pretrained_model = torch.load(cfg.checkpointing.pretrained_monodepth, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained monodepth: {cfg.checkpointing.pretrained_monodepth}"
                )
            )

        # load pretrained mvdepth
        if cfg.checkpointing.pretrained_mvdepth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_mvdepth, map_location='cpu')['model']

            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=False)
            print(
                cyan(
                    f"Loaded pretrained mvdepth: {cfg.checkpointing.pretrained_mvdepth}"
                )
            )
        
        # load full model
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
                )
            )

        # load pretrained depth
        if cfg.checkpointing.pretrained_depth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')['model']

            strict_load = True
            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
                )
            )
            
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        # load full model
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
                )
            )

        # load pretrained depth model only
        if cfg.checkpointing.pretrained_depth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')['model']

            strict_load = True
            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
                )
            )
            
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
