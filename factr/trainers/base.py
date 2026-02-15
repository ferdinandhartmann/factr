# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

TRAIN_LOG_FREQ, EVAL_LOG_FREQ = 100, 1


class RunningMean:
    def __init__(self, max_len=TRAIN_LOG_FREQ):
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
    def __init__(self, model, device_id, optim_builder, schedule_builder=None, checkpoint_dir="."):
        self.model, self.device_id = model, device_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_dir = self.checkpoint_dir / "rollout"
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
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
            optim_class = getattr(torch.optim, optimizer_type)
            optimizer_kwargs = dict(getattr(optim_builder, "optimizer_kwargs", {}))
            self.optim = optim_class(self.model.parameters(), **optimizer_kwargs)
        else:
            raise ValueError("Unsupported optim_builder format.")

        if not self._is_cuda_device(device_id) and hasattr(self.optim, "_cuda_graph_capture_health_check"):
            # CPU training can fail on some builds that probe CUDA stream capture unconditionally.
            self.optim._cuda_graph_capture_health_check = lambda: None

        self.schedule = None if schedule_builder is None else schedule_builder(self.optim)
        self._trackers = dict()
        self._is_train = True
        self.set_train()

    @staticmethod
    def _is_cuda_device(device_id):
        if isinstance(device_id, torch.device):
            return device_id.type == "cuda"
        if isinstance(device_id, str):
            return device_id.startswith("cuda")
        return isinstance(device_id, int)

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

        # Save current checkpoint
        current_ckpt = self.checkpoint_dir / f"ckpt_{global_step:06d}.ckpt"
        torch.save(save_dict, str(current_ckpt))

        # Remove old checkpoints, keeping only the 2 most recent
        ckpts = sorted(self.checkpoint_dir.glob("ckpt_*.ckpt"))
        for old_ckpt in ckpts[:-top_k]:  # Keep last 2 checkpoints
            old_ckpt.unlink()
        torch.save(save_dict, str(self.rollout_dir / "latest_ckpt.ckpt"))

    def load_checkpoint(self, load_path):
        load_path = Path(load_path)
        if not load_path.is_absolute():
            candidate = self.checkpoint_dir / load_path
            if candidate.exists():
                load_path = candidate

        load_dict = torch.load(str(load_path), weights_only=False)
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
                self._trackers[k] = RunningMean()

    def log(self, key, global_step, value):
        log_freq = TRAIN_LOG_FREQ if self._is_train else EVAL_LOG_FREQ
        key_prepend = "train/" if self._is_train else "eval/"
        key = key_prepend + key

        if key not in self._trackers:
            self._trackers[key] = RunningMean()

        tracker = self._trackers[key]
        tracker.append(value)

        if global_step % log_freq == 0 and wandb.run is not None:
            wandb.log({key: tracker.mean}, step=global_step)

    def set_device(self, device_id):
        # Move model to device
        self.model = self.model.to(device_id)

        # Enable multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)
