import torch

from factr.trainers.base import BaseTrainer


class GaussianObsPredictionTrainer(BaseTrainer):
    def training_step(self, batch, global_step):
        obs_window, action_chunk, target_obs, target_mask, stiffness_labels, goal_labels = batch
        obs_window = obs_window.to(self.device_id)
        action_chunk = action_chunk.to(self.device_id)
        target_obs = target_obs.to(self.device_id)
        target_mask = target_mask.to(self.device_id)
        stiffness_labels = stiffness_labels.to(self.device_id)
        goal_labels = goal_labels.to(self.device_id)

        output = self.model(
            obs_window=obs_window,
            action_chunk=action_chunk,
            stiffness_labels=stiffness_labels,
            goal_labels=goal_labels,
            target_obs=target_obs,
            target_mask=target_mask,
        )
        loss = output["loss"]

        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        tracking_error = model_ref.compute_tracking_error(output["mean"], action_chunk)
        tracking_l2_per_step = torch.linalg.norm(tracking_error, dim=-1)
        tracking_error_l2 = (tracking_l2_per_step * target_mask).sum() / torch.clamp(target_mask.sum(), min=1.0)
        pred_var_mean = output["var"].mean()

        if self.is_train:
            self.log("sample_mse", global_step, loss.item())
            self.log("mean_mse", global_step, output["mean_mse"].item())
            self.log("tracking_error_l2", global_step, tracking_error_l2.item())
            self.log("pred_var_mean", global_step, pred_var_mean.item())
            self.log("lr", global_step, self.lr)

        self.last_train_loss = loss.item()
        return loss
