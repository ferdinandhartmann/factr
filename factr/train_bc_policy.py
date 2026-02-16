# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
import traceback
import numbers
from copy import deepcopy
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import wandb
from factr import misc, transforms
from factr.trainers.base import TRAIN_LOG_FREQ

base_path = os.path.dirname(os.path.abspath(__file__))


def torch_fix_seed(seed: int = 42) -> None:
    """
    乱数を固定する関数.

    References
    ----------
    - https://qiita.com/north_redwing/items/1e153139125d37829d2d
    """
    random.seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _grad_l2_norm(model) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total += grad_norm * grad_norm
    return total**0.5


@hydra.main(version_base=None, config_path="cfg", config_name="train_bc_lowdim.yaml")
def train_bc(cfg: DictConfig):
    try:
        resume_model = misc.init_job(cfg)

        # # set random seeds for reproducibility
        # torch.manual_seed(cfg.seed)
        # np.random.seed(cfg.seed + 1)
        torch_fix_seed(cfg.seed)

        # save rollout config under checkpoint(run) directory
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        rollout_dir = run_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        with open(rollout_dir / "agent_config.yaml", "w") as f:
            inference_config = deepcopy(cfg.agent)
            if OmegaConf.select(inference_config, "features.restore_path", default=None) is not None:
                inference_config.features.restore_path = ""
            agent_yaml = OmegaConf.to_yaml(inference_config, resolve=True)
            f.write(agent_yaml)
        with open(rollout_dir / "exp_config.yaml", "w") as f:
            exp_yaml = OmegaConf.to_yaml(cfg)
            f.write(exp_yaml)
        rollout_cfg_path = Path(cfg.buffer_path).parent / "rollout_config.yaml"
        if rollout_cfg_path.exists():
            rollout_config = OmegaConf.load(rollout_cfg_path)
            with open(rollout_dir / "rollout_config.yaml", "w") as f:
                OmegaConf.save(rollout_config, f)

        exp_config_path = run_dir / "exp_config.yaml"
        print(f"Run directory: {run_dir}")
        print(f"Run exp config: {exp_config_path}")
        print(f"Rollout directory: {rollout_dir}")

        # build agent from hydra configs
        agent = hydra.utils.instantiate(cfg.agent)
        device_id = "cpu"
        if int(getattr(cfg, "devices", 1)) > 0 and torch.cuda.is_available():
            device_id = 0
        trainer = hydra.utils.instantiate(cfg.trainer, model=agent, device_id=device_id)
        if resume_model is not None and os.path.exists(resume_model):
            restored_step = trainer.load_checkpoint(resume_model)
            misc.GLOBAL_STEP = int(restored_step)
            print(f"Resumed checkpoint from {resume_model} at step {misc.GLOBAL_STEP}")

        # build task, replay buffer, and dataloader
        task = hydra.utils.instantiate(cfg.task, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        print(
            "Run config | "
            f"device={trainer.device_id} "
            f"train_buffer={cfg.buffer_path} "
            f"test_buffer={OmegaConf.select(cfg, 'test_buffer_path', default=cfg.buffer_path)} "
            f"batch_size={cfg.batch_size} "
            f"ac_chunk={cfg.ac_chunk} "
            f"obs_window={OmegaConf.select(cfg, 'obs_window', default='n/a')}"
        )
        if hasattr(task, "eval_plot_max_steps"):
            print(
                "Eval plot config | "
                f"max_steps={task.eval_plot_max_steps} "
                f"stride={task.eval_plot_prediction_stride} "
                f"num_samples={task.eval_plot_num_samples}"
            )

        # create a gpu train transform (if used)
        gpu_transform = (
            transforms.get_gpu_transform_by_name(cfg.train_transform) if "gpu" in cfg.train_transform else None
        )

        restore_path = OmegaConf.select(cfg, "agent.features.restore_path", default="")
        if resume_model is not None:
            restore_path = ""
        if restore_path and os.path.exists(restore_path):
            print(f"Manually loading weights from: {restore_path}")
            checkpoint = torch.load(restore_path, map_location="cpu")
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

            model_state = state_dict
            if hasattr(agent, "visual_features"):
                needs_prefix = len(model_state) > 0 and not any(k.startswith("visual_features.") for k in model_state)
                if needs_prefix:
                    model_state = {f"visual_features.0.{k}": v for k, v in model_state.items()}

            try:
                load_msg = agent.load_state_dict(model_state, strict=False)
                print(
                    f"Manual load finished. Missing={len(load_msg.missing_keys)} "
                    f"Unexpected={len(load_msg.unexpected_keys)}"
                )
            except RuntimeError as e:
                print(f"Load failed: {e}")

        trainer.set_train()
        train_iterator = iter(task.train_loader)
        for itr in (pbar := tqdm.tqdm(range(cfg.max_iterations), postfix=dict(Loss=None))):
            if itr < misc.GLOBAL_STEP:
                continue

            # infinitely sample batches until the train loop is finished
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(task.train_loader)
                batch = next(train_iterator)

            # handle the image transform on GPU if specified
            if gpu_transform is not None:
                if len(batch) == 4:
                    (imgs, obs), actions, mask, labels = batch
                    imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                    imgs = {k: gpu_transform(v) for k, v in imgs.items()}
                    batch = ((imgs, obs), actions, mask, labels)
                else:
                    (imgs, obs), actions, mask = batch
                    imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                    imgs = {k: gpu_transform(v) for k, v in imgs.items()}
                    batch = ((imgs, obs), actions, mask)

            trainer.optim.zero_grad()
            loss = trainer.training_step(batch, misc.GLOBAL_STEP)
            if loss.ndim > 0:
                loss = loss.mean()
            loss.backward()

            model_for_norm = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            grad_norm = _grad_l2_norm(model_for_norm)
            if wandb.run is not None and misc.GLOBAL_STEP % 1 == 0:
                wandb.log({"train/grad_norm": grad_norm}, step=misc.GLOBAL_STEP)

            trainer.optim.step()

            pbar.set_postfix(dict(Loss=loss.item()))

            if wandb.run is not None and misc.GLOBAL_STEP > 0 and misc.GLOBAL_STEP % TRAIN_LOG_FREQ == 0:
                payload = trainer.consume_wandb_payload(misc.GLOBAL_STEP)
                try:
                    payload["train/grad_norm"] = float(grad_norm)
                except Exception:
                    payload["train/grad_norm"] = grad_norm
                if payload:
                    pretty = " ".join(
                        f"{k}={float(v):.6g}" if isinstance(v, numbers.Number) else f"{k}={v}"
                        for k, v in sorted(payload.items())
                    )
                    print(f"[wandb] step={misc.GLOBAL_STEP} {pretty}")
            misc.GLOBAL_STEP += 1

            if misc.GLOBAL_STEP % cfg.schedule_freq == 0:
                trainer.step_schedule()

            if misc.GLOBAL_STEP % cfg.eval_freq == 0:
                print("\nEvaluating model...")
                trainer.set_eval()
                task.eval(trainer, misc.GLOBAL_STEP)
                trainer.set_train()

            if misc.GLOBAL_STEP >= cfg.max_iterations:
                trainer.save_checkpoint(misc.GLOBAL_STEP)
                return
            elif misc.GLOBAL_STEP % cfg.save_freq == 0:
                trainer.save_checkpoint(misc.GLOBAL_STEP)

    # gracefully handle and log errors
    except Exception:
        traceback.print_exc(file=open("exception.log", "w"))
        with open("exception.log", "r") as f:
            print(f.read())


if __name__ == "__main__":
    train_bc()
