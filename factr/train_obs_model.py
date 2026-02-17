import numbers
import os
import random
import traceback
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from factr import misc
from factr.trainers.base import TRAIN_LOG_FREQ


def torch_fix_seed(seed: int = 42) -> None:
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


@hydra.main(version_base=None, config_path="cfg", config_name="train_obs_pred_lowdim.yaml")
def train_obs_model(cfg: DictConfig):
    run_dir = None
    try:
        resume_model = misc.init_job(cfg)
        torch_fix_seed(cfg.seed)

        run_dir = Path(HydraConfig.get().runtime.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        rollout_dir = run_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        with open(rollout_dir / "obs_model_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        agent = hydra.utils.instantiate(cfg.agent)
        device_id = "cpu"
        if int(getattr(cfg, "devices", 1)) > 0 and torch.cuda.is_available():
            device_id = 0
        trainer = hydra.utils.instantiate(cfg.trainer, model=agent, device_id=device_id)

        if resume_model is not None and os.path.exists(resume_model):
            restored_step = trainer.load_checkpoint(resume_model)
            misc.GLOBAL_STEP = int(restored_step)
            print(f"Resumed checkpoint from {resume_model} at step {misc.GLOBAL_STEP}")

        task = hydra.utils.instantiate(cfg.task, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        print(
            "Run config | "
            f"device={trainer.device_id} "
            f"train_buffer={cfg.buffer_path} "
            f"test_buffer={cfg.test_buffer_path} "
            f"batch_size={cfg.batch_size} "
            f"obs_window={cfg.obs_window} "
            f"obs_input_dim={cfg.obs_input_dim} "
            f"obs_target_dim={cfg.obs_target_dim} "
            f"pred_horizon={cfg.pred_horizon} "
            f"pose_action_dim={cfg.pose_action_dim}"
        )

        trainer.set_train()
        train_iterator = iter(task.train_loader)
        for itr in (pbar := tqdm.tqdm(range(cfg.max_iterations), postfix=dict(Loss=None))):
            if itr < misc.GLOBAL_STEP:
                continue

            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(task.train_loader)
                batch = next(train_iterator)

            trainer.optim.zero_grad()
            loss = trainer.training_step(batch, misc.GLOBAL_STEP)
            if loss.ndim > 0:
                loss = loss.mean()
            loss.backward()

            model_for_norm = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            grad_norm = _grad_l2_norm(model_for_norm)
            if wandb.run is not None:
                wandb.log({"train/grad_norm": grad_norm}, step=misc.GLOBAL_STEP)

            trainer.optim.step()
            pbar.set_postfix(dict(Loss=loss.item()))

            if wandb.run is not None and misc.GLOBAL_STEP > 0 and misc.GLOBAL_STEP % TRAIN_LOG_FREQ == 0:
                payload = trainer.consume_wandb_payload(misc.GLOBAL_STEP)
                payload["train/grad_norm"] = float(grad_norm)
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
                generate_plots = misc.GLOBAL_STEP % cfg.eval_freq_plot == 0
                task.eval(trainer, misc.GLOBAL_STEP, generate_plots=generate_plots)
                trainer.set_train()

            if misc.GLOBAL_STEP >= cfg.max_iterations:
                trainer.save_checkpoint(misc.GLOBAL_STEP)
                return
            if misc.GLOBAL_STEP % cfg.save_freq == 0:
                trainer.save_checkpoint(misc.GLOBAL_STEP)

    except Exception:
        if run_dir is not None:
            exception_log_path = run_dir / "exception.log"
        else:
            exception_log_path = Path("exception.log")

        traceback.print_exc(file=open(exception_log_path, "w"))
        with open(exception_log_path, "r") as f:
            print(f.read())


if __name__ == "__main__":
    train_obs_model()
