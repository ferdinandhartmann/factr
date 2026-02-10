# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

import wandb
from factr.replay_buffer import IterableWrapper


def seed_worker(_worker_id: int) -> None:
    """
    DataLoaderのworkerの固定.

    Dataloaderの乱数固定にはgeneratorの固定も必要らしい
    """
    worker_seed = torch.initial_seed() % 2**32
    pl.seed_everything(worker_seed)


def _build_data_loader(buffer, batch_size, num_workers, is_train=False, shuffle=True):
    if is_train and not isinstance(buffer, IterableDataset):
        buffer = IterableWrapper(buffer)

    return DataLoader(
        buffer,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not isinstance(buffer, IterableDataset) and shuffle,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        # worker_init_fn=lambda _: np.random.seed(),
        worker_init_fn=seed_worker,
    )


class DefaultTask:
    def __init__(
        self,
        train_buffer,
        test_buffer,
        cam_indexes,
        n_cams,
        obs_dim,
        ac_dim,
        batch_size,
        num_workers,
        factr_baseline: bool = False,
    ):
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.train_loader = _build_data_loader(train_buffer, batch_size, num_workers, is_train=True)

        self.weights_history = []
        self.weights_steps = []
        # make sure no randomization and shuffling
        self.test_loader = DataLoader(
            test_buffer,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        self.factr_baseline = factr_baseline

    def eval(self, trainer, global_step):
        losses = []
        for batch in self.test_loader:
            with torch.no_grad():
                loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())

        mean_val_loss = np.mean(losses)
        print(f"Step: {global_step}\tVal Loss: {mean_val_loss:.4f}")
        if wandb.run is not None:
            wandb.log({"eval/task_loss": mean_val_loss}, step=global_step)


class BCTask(DefaultTask):
    def eval(self, trainer, global_step):
        losses = []
        action_l2, action_lsig = [], []
        l2_per_joint_all = []
        accuracy_list = []  # ★ 初期化

        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        was_training = model.training
        model.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                # 1. データ受け取り
                (imgs, obs), actions, mask, labels = batch

                # 2. GPU転送
                imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                obs, actions, mask, labels = [ar.to(trainer.device_id) for ar in (obs, actions, mask, labels)]

                ac_flat = actions.reshape((actions.shape[0], -1))
                mask_flat = mask.reshape((mask.shape[0], -1))

                output_dict = model(imgs, obs, ac_flat, mask_flat, class_labels=labels)

                losses.append(output_dict["l1_loss"].item())

                if getattr(model, "factr_baseline", False):
                    pred_actions = model.get_actions_base(imgs, obs)
                else:
                    pred_actions = model.get_actions_prior(imgs, obs)
                    pred_actions = pred_actions.squeeze(1)

                l2_delta = torch.square(mask * (pred_actions - actions))
                l2_delta = l2_delta.sum((1, 2)) / mask.sum((1, 2))
                action_l2.append(l2_delta.mean().item())

                l2_per_joint = (mask * (pred_actions - actions) ** 2).sum(1) / mask.sum(1)
                l2_per_joint_all.append(l2_per_joint.mean(0).cpu().numpy())

                lsig = torch.logical_or(
                    torch.logical_and(actions > 0, pred_actions <= 0),
                    torch.logical_and(actions <= 0, pred_actions > 0),
                )
                lsig = (lsig.float() * mask).sum((1, 2)) / mask.sum((1, 2))
                action_lsig.append(lsig.mean().item())

                if output_dict.get("logits") is not None:
                    logits = output_dict["logits"]
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == labels).float().mean().item()
                    accuracy_list.append(acc)

        if was_training:
            model.train()

        mean_val_loss = np.mean(losses)
        ac_l2 = np.mean(action_l2)
        ac_lsig = np.mean(action_lsig)
        l2_per_joint_mean = np.mean(np.stack(l2_per_joint_all, axis=0), axis=0)
        l2_per_joint_mean_7d = l2_per_joint_mean

        print(f"Step: {global_step}\tVal Loss(L1): {mean_val_loss:.4f}\tAction L2: {ac_l2:.3f}")

        if wandb.run is not None:
            for i, v in enumerate(l2_per_joint_mean_7d):
                wandb.log({f"eval/joint{i + 1}_l2": v}, step=global_step)

            log_dict = {
                "eval/task_loss": mean_val_loss,
                "eval/action_l2": ac_l2,
                "eval/action_lsig": ac_lsig,
            }

            if accuracy_list:
                log_dict["eval/classification_accuracy"] = np.mean(accuracy_list)

            wandb.log(log_dict, step=global_step)
