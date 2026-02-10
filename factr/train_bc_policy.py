# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
import traceback
from copy import deepcopy
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

from factr import misc, transforms

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


@hydra.main(config_path="cfg", config_name="train_bc.yaml")
def train_bc(cfg: DictConfig):
    try:
        resume_model = misc.init_job(cfg)

        # # set random seeds for reproducibility
        # torch.manual_seed(cfg.seed)
        # np.random.seed(cfg.seed + 1)
        torch_fix_seed(cfg.seed)

        # save rollout config
        rollout_dir = Path("rollout")
        if not rollout_dir.exists():
            rollout_dir.mkdir()
            with open(rollout_dir / "agent_config.yaml", "w") as f:
                inference_config = deepcopy(cfg.agent)
                inference_config.features.restore_path = ""
                agent_yaml = OmegaConf.to_yaml(inference_config, resolve=True)
                f.write(agent_yaml)
            with open(rollout_dir / "exp_config.yaml", "w") as f:
                exp_yaml = OmegaConf.to_yaml(cfg)
                f.write(exp_yaml)
            rollout_config = OmegaConf.load(Path(cfg.buffer_path).parent / "rollout_config.yaml")
            with open(rollout_dir / "rollout_config.yaml", "w") as f:
                OmegaConf.save(rollout_config, f)

        # build agent from hydra configs
        agent = hydra.utils.instantiate(cfg.agent)
        trainer = hydra.utils.instantiate(cfg.trainer, model=agent, device_id=0)

        # build task, replay buffer, and dataloader
        task = hydra.utils.instantiate(cfg.task, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        # create a gpu train transform (if used)
        gpu_transform = (
            transforms.get_gpu_transform_by_name(cfg.train_transform) if "gpu" in cfg.train_transform else None
        )

        # restore/save the model as required
        # 1. 自動ロードを無効化
        restore_path = cfg.agent.features.restore_path
        cfg.agent.features.restore_path = ""
        print(f"Disabled auto-loading. Manual load path: {restore_path}")

        # 2. Agent初期化
        agent = hydra.utils.instantiate(cfg.agent)

        # 3. シンプルに手動ロード（ここを書き換えてください！）
        # 3. 手動で重みをロード
        if restore_path and os.path.exists(restore_path):
            print(f"Manually loading features from: {restore_path}")
            checkpoint = torch.load(restore_path, map_location="cpu")  # 変数名をcheckpointに変更

            # === 【追加修正】 "model" という箱に入っている場合、中身を取り出す ===
            if "model" in checkpoint:
                print("Found 'model' key. Unwrapping...")
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            # =============================================================

            # 名前の変換処理（ここは前回と同じ）
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = f"visual_features.0.{k}"
                new_state_dict[new_key] = v

            try:
                # new_state_dict をロード
                msg = agent.load_state_dict(new_state_dict, strict=False)
                # print(f"Load result: {msg}")

                # 【確認】今回は missing_keys が激減するはずです
                # もし visual_features 関連が消えていれば成功です
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
                (imgs, obs), actions, mask = batch
                imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                imgs = {k: gpu_transform(v) for k, v in imgs.items()}
                batch = ((imgs, obs), actions, mask)

            trainer.optim.zero_grad()
            loss = trainer.training_step(batch, misc.GLOBAL_STEP)
            if loss.ndim > 0:
                loss = loss.mean()
            loss.backward()
            trainer.optim.step()

            pbar.set_postfix(dict(Loss=loss.item()))
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
