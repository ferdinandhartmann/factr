# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

TRAIN_LOG_FREQ, EVAL_LOG_FREQ = 100, 1


class RunningMean:
    def __init__(self, max_len=100):
        self._values = []
        self._ctr, self._max_len = 0, max_len

    def append(self, item):
        self._ctr = (self._ctr + 1) % self._max_len
        if len(self._values) < self._max_len:
            self._values.append(item)
        else:
            self._values[self._ctr] = item

    @property
    def mean(self):
        if len(self._values) == 0:
            raise ValueError
        return np.mean(self._values)


class BaseTrainer(ABC):
    def __init__(self, model, device_id, optim_builder, schedule_builder=None, train_log_freq=TRAIN_LOG_FREQ):
        self.model, self.device_id = model, device_id
        self.train_log_freq = int(train_log_freq)
        self.set_device(device_id)
        optimizer_type = getattr(optim_builder, "optimizer_type", None)

        if optimizer_type == "custom_mae_lrd_adamW":
            from factr.trainers import lrd

            """optimizer from mae codebase """
            param_groups = lrd.param_groups_lrd(
                model,
                optim_builder.optimizer_kwargs.weight_decay,
                # no_weight_decay_list=model.no_weight_decay(),
                layer_decay=optim_builder.optimizer_kwargs.layer_decay,
            )
            self.optim = torch.optim.AdamW(param_groups, lr=optim_builder.optimizer_kwargs.lr)
        elif callable(optim_builder):
            self.optim = optim_builder(self.model.parameters())
        elif optimizer_type is not None:
            optimizer_class = getattr(torch.optim, optimizer_type)
            optimizer_kwargs = dict(getattr(optim_builder, "optimizer_kwargs", {}))
            self.optim = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        else:
            raise ValueError("Unsupported optim_builder format.")

        self.schedule = None if schedule_builder is None else schedule_builder(self.optim)
        self._trackers = dict()
        self._is_train = True
        self._wandb_last_step = None
        self._wandb_last_payload = {}
        self.set_train()

    @abstractmethod
    def training_step(self, batch_input, global_step):
        pass

    @property
    def lr(self):
        if self.schedule is None:
            return self.optim.param_groups[0]["lr"]
        return self.schedule.get_last_lr()[0]

    def step_schedule(self):
        if self.schedule is None:
            return
        self.schedule.step()

    def save_checkpoint(self, global_step, top_k=2):
        model = self.model
        model_weights = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        schedule_state = dict() if self.schedule is None else self.schedule.state_dict()
        save_dict = dict(
            model=model_weights,
            optim=self.optim.state_dict(),
            schedule=schedule_state,
            global_step=global_step,
        )

        run_dir = self._get_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save current checkpoint
        current_ckpt = run_dir / f"ckpt_{global_step:06d}.ckpt"
        torch.save(save_dict, current_ckpt)

        # Remove old checkpoints, keeping only the 2 most recent
        ckpts = sorted(run_dir.glob("ckpt_*.ckpt"))
        for old_ckpt in ckpts[:-top_k]:  # Keep last 2 checkpoints
            old_ckpt.unlink()

        rollout_dir = run_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, rollout_dir / "latest_ckpt.ckpt")

    def load_checkpoint(self, load_path):
        load_dict = torch.load(load_path, weights_only=False)
        model = self.model
        model = model.module if isinstance(model, DDP) else model
        model.load_state_dict(load_dict["model"])

        self.optim.load_state_dict(load_dict["optim"])
        if self.schedule is not None:
            self.schedule.load_state_dict(load_dict["schedule"])

        return load_dict["global_step"]

    def _load_callback(self, load_path, load_dict):
        pass

    @property
    def is_train(self):
        return self._is_train

    def set_train(self):
        self._is_train = True
        self.model = self.model.train()

    def set_eval(self):
        self._is_train = False
        self.model = self.model.eval()

        # reset running mean for eval trackers
        for k in self._trackers:
            if "eval/" in k:
                self._trackers[k] = RunningMean(max_len=EVAL_LOG_FREQ)

    def log(self, key, global_step, value):
        log_freq = self.train_log_freq if self._is_train else EVAL_LOG_FREQ
        key_prepend = "train/" if self._is_train else "eval/"
        key = key_prepend + key

        if key not in self._trackers:
            max_len = self.train_log_freq if self._is_train else EVAL_LOG_FREQ
            self._trackers[key] = RunningMean(max_len=max_len)

        tracker = self._trackers[key]
        tracker.append(value)

        if global_step % log_freq == 0 and wandb.run is not None:
            mean_val = float(tracker.mean)
            wandb.log({key: mean_val}, step=global_step)

            # Cache the exact payload we sent to wandb so callers can optionally
            # print an aggregated line to the terminal.
            if self._wandb_last_step != global_step:
                self._wandb_last_step = global_step
                self._wandb_last_payload = {}
            self._wandb_last_payload[key] = mean_val

    def consume_wandb_payload(self, global_step: int) -> dict:
        """Return (and clear) the last cached wandb payload for `global_step`."""
        if self._wandb_last_step != global_step:
            return {}
        payload = dict(self._wandb_last_payload)
        self._wandb_last_payload = {}
        return payload

    def set_device(self, device_id):
        # Move model to device
        self.model = self.model.to(device_id)

        # Enable multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)

    @staticmethod
    def _get_run_dir() -> Path:
        try:
            return Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            return Path.cwd()
