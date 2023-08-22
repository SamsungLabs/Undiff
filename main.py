import os

import hydra
import torch

import improved_diffusion
from improved_diffusion.tasks import TaskType


def instantiate_model_and_diffusion(cfg, device):
    model = hydra.utils.instantiate(cfg.model.model)

    for param in model.parameters():
        param.requires_grad = False

    # load checkpoint
    pl_ckpt = torch.load(cfg.model.ckpt_path, map_location="cpu")
    model_state = improved_diffusion.remove_prefix_from_state_dict(
        pl_ckpt["state_dict"], j=1
    )

    # load ema
    if cfg.use_ema:
        ema_params = pl_ckpt["ema_params"][0]
        model_state = improved_diffusion.create_state_dict_from_ema(
            model_state, model, ema_params
        )

    # load state_dict
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # define diffusion
    diffusion = hydra.utils.call(cfg.diffusion)

    return model, diffusion


@hydra.main(config_path="configs", config_name="inference_cfg", version_base=None)
def inference(cfg):
    DEVICE = cfg.device

    metrics_list = [
        hydra.utils.instantiate(cfg.task.metrics[metric], device=DEVICE)
        for metric in cfg.task.metrics
    ]

    task = hydra.utils.instantiate(
        cfg.task.name, output_dir=cfg.output_dir, metrics=metrics_list
    )

    model, diffusion = instantiate_model_and_diffusion(cfg, DEVICE)

    files_or_num = (
        list(map(lambda x: os.path.join(cfg.audio_dir, x), os.listdir(cfg.audio_dir)))
        if task.task_type != TaskType.UNCONDITIONAL
        else cfg.audio_dir
    )

    task.inference(
        files_or_num, model, diffusion, cfg.sampling_rate, cfg.segment_size, DEVICE
    )


if __name__ == "__main__":
    inference()
