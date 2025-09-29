# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9581))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

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
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model="all",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
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
        )
    )

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        None, # for zpressor
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        if cfg.checkpointing.pretrained_model:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model)
            if "state_dict" in pretrained_model:
                pretrained_model = pretrained_model["state_dict"]
            # If num_context_views is not 4, need to modify encoder.epipolar_transformer.view_embeddings.weight, replicate it num_context_views // 4 times and concatenate to load into model_wrapper.encoder.epipolar_transformer.view_embeddings.weight
            if "encoder.epipolar_transformer.view_embeddings.weight" in pretrained_model:
                old_weight = pretrained_model["encoder.epipolar_transformer.view_embeddings.weight"]
                if old_weight.shape[0] != model_wrapper.encoder.epipolar_transformer.view_embeddings.weight.shape[0]:
                    old_shape = old_weight.shape[0]
                    new_shape = model_wrapper.encoder.epipolar_transformer.view_embeddings.weight.shape[0]
                    print(f"old_shape: {old_shape}, new_shape: {new_shape}")
                    new_weight = model_wrapper.encoder.epipolar_transformer.view_embeddings.weight.clone()
                    
                    # Fix calculation formula: ensure indices don't exceed range
                    # New weight[n] = old_weight[n % old_shape]
                    for n in range(new_shape):
                        new_weight[n] = old_weight[n % old_shape]
                    
                    # Update model weights
                    model_wrapper.encoder.epipolar_transformer.view_embeddings.weight.data = new_weight
                    
                    # Remove this key from pretrained model for subsequent loading of other parameters
                    del pretrained_model["encoder.epipolar_transformer.view_embeddings.weight"]
            
            # Load other parameters
            model_wrapper.load_state_dict(pretrained_model, strict=False)
        
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
