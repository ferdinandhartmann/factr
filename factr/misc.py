# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools
import os
import signal
import sys
from pathlib import Path

import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import wandb
from factr.transforms import get_transform_by_name

OmegaConf.register_new_resolver("env", lambda x: os.environ[x])
OmegaConf.register_new_resolver("base", lambda: os.path.dirname(os.path.abspath(__file__)))
OmegaConf.register_new_resolver("transform", lambda name: get_transform_by_name(name))
try:
    OmegaConf.register_new_resolver("mult", lambda x, y: int(x) * int(y))
except ValueError:
    pass
OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))
OmegaConf.register_new_resolver("index", lambda arr, idx: arr[idx])
OmegaConf.register_new_resolver("len", lambda arr: len(arr))


GLOBAL_STEP = 0
REQUEUE_CAUGHT = False


def _signal_helper(signal, frame, prior_handler, trainer):
    global REQUEUE_CAUGHT, GLOBAL_STEP
    REQUEUE_CAUGHT = True

    # save train checkpoint
    print(f"Caught requeue signal at step: {GLOBAL_STEP}")
    trainer.save_checkpoint(GLOBAL_STEP)

    # return back to submitit handler if it exists
    if callable(prior_handler):
        return prior_handler(signal, frame)
    return sys.exit(-1)


def set_checkpoint_handler(trainer):
    global REQUEUE_CAUGHT
    REQUEUE_CAUGHT = False
    prior_handler = signal.getsignal(signal.SIGUSR2)
    handler = functools.partial(
        _signal_helper,
        prior_handler=prior_handler,
        trainer=trainer,
    )
    signal.signal(signal.SIGUSR2, handler)


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    if wandb_cfg.debug:
        return "null_id"
    try:
        job_id = HydraConfig().get().job.num
        override_dirname = HydraConfig().get().job.override_dirname
        name = f"{wandb_cfg.sweep_name_prefix}-{job_id}"
        notes = f"{override_dirname}"
    except:
        name, notes = wandb_cfg.name, None

    wandb_run = wandb.init(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        # entity=wandb_cfg.entity,
        config=job_config,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )
    return wandb_run.id


def _get_run_dir() -> Path:
    try:
        output_dir = HydraConfig.get().runtime.output_dir
        return Path(output_dir)
    except Exception:
        return Path.cwd()


def init_job(cfg):
    cfg_yaml = OmegaConf.to_yaml(cfg)
    params = yaml.safe_load(cfg_yaml)
    exp_name = str(params.get("exp_name", ""))
    run_dir = _get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    exp_config_path = run_dir / "exp_config.yaml"
    latest_ckpt = run_dir / "rollout" / "latest_ckpt.ckpt"

    should_resume = False
    if exp_config_path.exists():
        with exp_config_path.open("r") as f:
            old_config = yaml.safe_load(f)
        old_params = old_config.get("params", {}) if isinstance(old_config, dict) else {}
        old_exp_name = str(old_params.get("exp_name", ""))

        explicit_resume = OmegaConf.select(cfg, "resume", default=None)
        if explicit_resume is None:
            same_run_name = (exp_name != "") and (old_exp_name == exp_name)
            should_resume = same_run_name and latest_ckpt.exists()
        else:
            should_resume = bool(explicit_resume) and latest_ckpt.exists()

        if should_resume:
            create_wandb_run(cfg.wandb, old_params, old_config.get("wandb_id"))
            return str(latest_ckpt)

    wandb_id = create_wandb_run(cfg.wandb, params)
    save_dict = dict(wandb_id=wandb_id, params=params)
    with exp_config_path.open("w") as f:
        yaml.dump(save_dict, f)
    rollout_dir = run_dir / "rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    with (rollout_dir / "exp_config.yaml").open("w") as f:
        yaml.dump(save_dict, f)
    return None
