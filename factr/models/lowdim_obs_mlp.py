import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LowdimGaussianObsMLP(nn.Module):
    requires_stiffness_label = True

    def __init__(
        self,
        obs_window=8,
        obs_input_dim=21,
        predict_obs_dim=21,
        pred_horizon=30,
        pose_action_dim=9,
        stiffness_classes=3,
        goal_classes=4,
        hidden_dim=384,
        num_layers=3,
        dropout=0.1,
        min_var=1e-4,
    ):
        super().__init__()
        self.obs_window = int(obs_window)
        self.obs_input_dim = int(obs_input_dim)
        self.predict_obs_dim = int(predict_obs_dim)
        self.pred_horizon = int(pred_horizon)
        self.pose_action_dim = int(pose_action_dim)
        self.stiffness_classes = int(stiffness_classes)
        self.goal_classes = int(goal_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.min_var = float(min_var)

        if not (2 <= self.num_layers <= 4):
            raise ValueError(f"num_layers must be in [2, 4], got {self.num_layers}.")
        if self.pred_horizon < 1:
            raise ValueError(f"pred_horizon must be >= 1, got {self.pred_horizon}.")
        if self.predict_obs_dim < self.pose_action_dim:
            raise ValueError(
                f"predict_obs_dim={self.predict_obs_dim} must be >= pose_action_dim={self.pose_action_dim}."
            )

        input_dim = (
            (self.obs_window * self.obs_input_dim)
            + (self.pred_horizon * self.pose_action_dim)
            + self.stiffness_classes
            + self.goal_classes
        )
        layers = []
        current_dim = input_dim
        for _ in range(self.num_layers):
            next_dim = self.hidden_dim
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, self.pred_horizon * self.predict_obs_dim * 2))
        self.backbone = nn.Sequential(*layers)

    def _normalize_labels(self, labels, num_classes, batch_size, device):
        labels = labels.to(device=device).long().view(-1)
        if labels.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} labels, got {labels.shape[0]}.")
        # Buffer labels are often 1-indexed; shift to 0-index before one-hot.
        if torch.min(labels) >= 1:
            labels = labels - 1
        labels = labels.clamp(min=0, max=num_classes - 1)
        return labels

    def normalize_goal_labels(self, goal_labels, batch_size=None, device=None):
        if batch_size is None:
            batch_size = int(goal_labels.numel())
        if device is None:
            device = goal_labels.device
        return self._normalize_labels(goal_labels, self.goal_classes, int(batch_size), device)

    def _build_features(self, obs_window, action_chunk, stiffness_labels, goal_labels):
        if obs_window.ndim != 3:
            raise ValueError(f"Expected obs_window shape (B, W, D), got {tuple(obs_window.shape)}.")
        if action_chunk.ndim != 3:
            raise ValueError(f"Expected action_chunk shape (B, H, A), got {tuple(action_chunk.shape)}.")

        batch_size, window, obs_dim = obs_window.shape
        if window != self.obs_window:
            raise ValueError(f"Expected obs_window={self.obs_window}, got {window}.")
        if obs_dim != self.obs_input_dim:
            raise ValueError(f"Expected obs_input_dim={self.obs_input_dim}, got {obs_dim}.")
        if (
            action_chunk.shape[0] != batch_size
            or action_chunk.shape[1] != self.pred_horizon
            or action_chunk.shape[2] != self.pose_action_dim
        ):
            raise ValueError(
                f"Expected action_chunk shape ({batch_size}, {self.pred_horizon}, {self.pose_action_dim}), "
                f"got {tuple(action_chunk.shape)}."
            )

        stiffness_idx = self._normalize_labels(stiffness_labels, self.stiffness_classes, batch_size, obs_window.device)
        goal_idx = self._normalize_labels(goal_labels, self.goal_classes, batch_size, obs_window.device)
        stiffness_onehot = F.one_hot(stiffness_idx, num_classes=self.stiffness_classes).float()
        goal_onehot = F.one_hot(goal_idx, num_classes=self.goal_classes).float()

        flat_obs = obs_window.reshape(batch_size, self.obs_window * self.obs_input_dim)
        flat_action = action_chunk.reshape(batch_size, self.pred_horizon * self.pose_action_dim)
        return torch.cat([flat_obs, flat_action, stiffness_onehot, goal_onehot], dim=-1)

    def _reshape_target_mask(self, target_mask, reference):
        if target_mask is None:
            return torch.ones(reference.shape[:2], device=reference.device, dtype=reference.dtype)
        if target_mask.ndim != 2:
            raise ValueError(f"Expected target_mask shape (B, H), got {tuple(target_mask.shape)}.")
        if target_mask.shape[:2] != reference.shape[:2]:
            raise ValueError(
                f"Expected target_mask shape {tuple(reference.shape[:2])}, got {tuple(target_mask.shape)}."
            )
        return target_mask.to(device=reference.device, dtype=reference.dtype)

    def _predict_distribution(self, obs_window, action_chunk, stiffness_labels, goal_labels):
        features = self._build_features(obs_window, action_chunk, stiffness_labels, goal_labels)
        out = self.backbone(features).view(features.shape[0], self.pred_horizon, self.predict_obs_dim, 2)
        mean = out[..., 0]
        var = F.softplus(out[..., 1]) + self.min_var
        std = torch.sqrt(var)
        return out, mean, var, std

    @staticmethod
    def sample_from_gaussian(mean, std, num_samples=1):
        n = int(num_samples)
        if n < 1:
            raise ValueError(f"num_samples must be >= 1, got {n}.")
        dist = Normal(loc=mean, scale=std.clamp_min(1e-8))
        if n == 1:
            return dist.rsample()
        return dist.rsample((n,))

    def _expand_batch_for_all_goals(self, obs_window, action_chunk, stiffness_labels):
        if obs_window.ndim != 3:
            raise ValueError(f"Expected obs_window shape (B, W, D), got {tuple(obs_window.shape)}.")
        if action_chunk.ndim != 3:
            raise ValueError(f"Expected action_chunk shape (B, H, A), got {tuple(action_chunk.shape)}.")

        batch_size, obs_window_len, obs_dim = map(int, obs_window.shape)
        if obs_window_len != self.obs_window:
            raise ValueError(f"Expected obs_window={self.obs_window}, got {obs_window_len}.")
        if obs_dim != self.obs_input_dim:
            raise ValueError(f"Expected obs_input_dim={self.obs_input_dim}, got {obs_dim}.")

        _, pred_horizon, action_dim = map(int, action_chunk.shape)
        if int(action_chunk.shape[0]) != batch_size:
            raise ValueError(
                f"obs_window/action_chunk batch mismatch: {int(obs_window.shape[0])} vs {int(action_chunk.shape[0])}."
            )
        if pred_horizon != self.pred_horizon or action_dim != self.pose_action_dim:
            raise ValueError(
                f"Expected action_chunk shape ({batch_size}, {self.pred_horizon}, {self.pose_action_dim}), "
                f"got {tuple(action_chunk.shape)}."
            )

        stiffness_labels = stiffness_labels.to(device=obs_window.device).long().view(-1)
        if int(stiffness_labels.shape[0]) != batch_size:
            raise ValueError(f"Expected {batch_size} stiffness labels, got {int(stiffness_labels.shape[0])}.")

        goals_1_indexed = (
            torch.arange(1, self.goal_classes + 1, device=obs_window.device, dtype=torch.long)
            .view(1, self.goal_classes)
            .expand(batch_size, self.goal_classes)
        )

        obs_all = (
            obs_window.unsqueeze(1)
            .expand(batch_size, self.goal_classes, obs_window_len, obs_dim)
            .reshape(batch_size * self.goal_classes, obs_window_len, obs_dim)
        )
        action_all = (
            action_chunk.unsqueeze(1)
            .expand(batch_size, self.goal_classes, pred_horizon, action_dim)
            .reshape(batch_size * self.goal_classes, pred_horizon, action_dim)
        )
        stiffness_all = (
            stiffness_labels.view(batch_size, 1).expand(batch_size, self.goal_classes).reshape(batch_size * self.goal_classes)
        )
        goals_all = goals_1_indexed.reshape(batch_size * self.goal_classes)
        return obs_all, action_all, stiffness_all, goals_all, goals_1_indexed

    def gaussian_log_likelihood(self, target_obs, mean, std, target_mask=None):
        if mean.ndim != 4:
            raise ValueError(f"Expected mean shape (B, G, H, D), got {tuple(mean.shape)}.")
        if std.shape != mean.shape:
            raise ValueError(f"Expected std shape {tuple(mean.shape)}, got {tuple(std.shape)}.")

        batch_size, goal_classes, horizon, pred_dim = mean.shape
        if goal_classes != self.goal_classes:
            raise ValueError(f"Expected goal axis size {self.goal_classes}, got {goal_classes}.")
        if target_obs.ndim != 3:
            raise ValueError(f"Expected target_obs shape (B, H, D), got {tuple(target_obs.shape)}.")
        if tuple(target_obs.shape) != (batch_size, horizon, pred_dim):
            raise ValueError(
                f"Expected target_obs shape ({batch_size}, {horizon}, {pred_dim}), got {tuple(target_obs.shape)}."
            )

        mask = self._reshape_target_mask(target_mask, reference=target_obs)
        target_expanded = target_obs.unsqueeze(1).expand(batch_size, goal_classes, horizon, pred_dim)
        dist = Normal(loc=mean, scale=std.clamp_min(1e-8))
        log_prob_per_dim = dist.log_prob(target_expanded)
        log_likelihood_per_timestep = log_prob_per_dim.sum(dim=-1)
        log_likelihood_per_timestep = log_likelihood_per_timestep * mask.unsqueeze(1)
        log_likelihood_per_goal = log_likelihood_per_timestep.sum(dim=-1)
        return {
            "log_prob_per_dim": log_prob_per_dim,
            "log_likelihood_per_timestep": log_likelihood_per_timestep,
            "log_likelihood_per_goal": log_likelihood_per_goal,
            "target_mask": mask,
        }

    def _normalize_goal_prior(self, goal_prior, batch_size, device, dtype):
        if goal_prior is None:
            return torch.full(
                (batch_size, self.goal_classes),
                1.0 / float(self.goal_classes),
                device=device,
                dtype=dtype,
            )

        prior = goal_prior.to(device=device, dtype=dtype)
        if prior.ndim == 1:
            if int(prior.shape[0]) != self.goal_classes:
                raise ValueError(f"Expected goal_prior shape ({self.goal_classes},), got {tuple(prior.shape)}.")
            prior = prior.view(1, self.goal_classes).expand(batch_size, self.goal_classes)
        elif prior.ndim == 2:
            if tuple(prior.shape) != (batch_size, self.goal_classes):
                raise ValueError(
                    f"Expected goal_prior shape ({batch_size}, {self.goal_classes}), got {tuple(prior.shape)}."
                )
        else:
            raise ValueError(
                f"Expected goal_prior shape ({self.goal_classes},) or ({batch_size}, {self.goal_classes}), "
                f"got {tuple(prior.shape)}."
            )
        prior = prior.clamp_min(1e-8)
        prior = prior / torch.clamp(prior.sum(dim=-1, keepdim=True), min=1e-8)
        return prior

    def compute_goal_posterior(self, log_likelihood_per_goal, goal_prior=None):
        if log_likelihood_per_goal.ndim != 2:
            raise ValueError(
                f"Expected log_likelihood_per_goal shape (B, G), got {tuple(log_likelihood_per_goal.shape)}."
            )
        batch_size, goal_classes = log_likelihood_per_goal.shape
        if goal_classes != self.goal_classes:
            raise ValueError(f"Expected G={self.goal_classes}, got {goal_classes}.")

        prior = self._normalize_goal_prior(
            goal_prior,
            batch_size=batch_size,
            device=log_likelihood_per_goal.device,
            dtype=log_likelihood_per_goal.dtype,
        )
        log_prior = torch.log(prior)
        log_unnormalized = log_prior + log_likelihood_per_goal
        log_posterior = log_unnormalized - torch.logsumexp(log_unnormalized, dim=-1, keepdim=True)
        return torch.exp(log_posterior), log_posterior

    def compute_goal_posterior_over_time(self, log_likelihood_per_timestep, goal_prior=None):
        if log_likelihood_per_timestep.ndim != 3:
            raise ValueError(
                "Expected log_likelihood_per_timestep shape (B, G, H), "
                f"got {tuple(log_likelihood_per_timestep.shape)}."
            )
        batch_size, goal_classes, _ = log_likelihood_per_timestep.shape
        if goal_classes != self.goal_classes:
            raise ValueError(f"Expected G={self.goal_classes}, got {goal_classes}.")

        prior = self._normalize_goal_prior(
            goal_prior,
            batch_size=batch_size,
            device=log_likelihood_per_timestep.device,
            dtype=log_likelihood_per_timestep.dtype,
        )
        cumulative_log_likelihood = torch.cumsum(log_likelihood_per_timestep.permute(0, 2, 1), dim=1)
        log_post_unnorm = torch.log(prior).unsqueeze(1) + cumulative_log_likelihood
        log_post = log_post_unnorm - torch.logsumexp(log_post_unnorm, dim=-1, keepdim=True)
        return torch.exp(log_post), log_post

    def infer_goals(
        self,
        obs_window,
        action_chunk,
        stiffness_labels,
        target_obs,
        target_mask=None,
        goal_prior=None,
        num_goal_samples=1,
    ):
        if target_obs is None:
            raise ValueError("target_obs is required for goal inference.")
        if target_obs.ndim != 3:
            raise ValueError(f"Expected target_obs shape (B, H, D), got {tuple(target_obs.shape)}.")

        batch_size = int(obs_window.shape[0])
        obs_all, action_all, stiffness_all, goals_all, goal_grid = self._expand_batch_for_all_goals(
            obs_window=obs_window,
            action_chunk=action_chunk,
            stiffness_labels=stiffness_labels,
        )
        dist_params_all, mean_all, var_all, std_all = self._predict_distribution(
            obs_window=obs_all,
            action_chunk=action_all,
            stiffness_labels=stiffness_all,
            goal_labels=goals_all,
        )

        mean = mean_all.view(batch_size, self.goal_classes, self.pred_horizon, self.predict_obs_dim)
        var = var_all.view(batch_size, self.goal_classes, self.pred_horizon, self.predict_obs_dim)
        std = std_all.view(batch_size, self.goal_classes, self.pred_horizon, self.predict_obs_dim)
        dist_params = dist_params_all.view(batch_size, self.goal_classes, self.pred_horizon, self.predict_obs_dim, 2)

        likelihood_out = self.gaussian_log_likelihood(
            target_obs=target_obs,
            mean=mean,
            std=std,
            target_mask=target_mask,
        )
        goal_posterior, log_goal_posterior = self.compute_goal_posterior(
            likelihood_out["log_likelihood_per_goal"],
            goal_prior=goal_prior,
        )
        goal_posterior_over_time, log_goal_posterior_over_time = self.compute_goal_posterior_over_time(
            likelihood_out["log_likelihood_per_timestep"],
            goal_prior=goal_prior,
        )

        num_goal_samples = int(num_goal_samples)
        sampled_single = self.sample_from_gaussian(mean, std, num_samples=1)
        sampled_multi = self.sample_from_gaussian(mean, std, num_samples=num_goal_samples) if num_goal_samples > 1 else None

        return {
            "goal_labels": goal_grid,
            "dist_params": dist_params,
            "mean": mean,
            "var": var,
            "std": std,
            "sample": sampled_single,
            "samples": sampled_multi,
            "log_prob_per_dim": likelihood_out["log_prob_per_dim"],
            "log_likelihood_per_timestep": likelihood_out["log_likelihood_per_timestep"],
            "log_likelihood_per_goal": likelihood_out["log_likelihood_per_goal"],
            "target_mask": likelihood_out["target_mask"],
            "goal_posterior": goal_posterior,
            "log_goal_posterior": log_goal_posterior,
            "goal_posterior_over_time": goal_posterior_over_time,
            "log_goal_posterior_over_time": log_goal_posterior_over_time,
            "predicted_goal": torch.argmax(goal_posterior, dim=-1) + 1,
        }

    def forward(self, obs_window, action_chunk, stiffness_labels, goal_labels, target_obs=None, target_mask=None):
        out, mean, var, std = self._predict_distribution(obs_window, action_chunk, stiffness_labels, goal_labels)
        sample = self.sample_from_gaussian(mean, std, num_samples=1)

        result = {"dist_params": out, "mean": mean, "var": var, "std": std, "sample": sample}
        if target_obs is not None:
            if target_obs.ndim != 3 or target_obs.shape != mean.shape:
                raise ValueError(
                    f"Expected target_obs shape {tuple(mean.shape)}, got {tuple(target_obs.shape)}."
                )
            mask = self._reshape_target_mask(target_mask, reference=mean)
            mask_expanded = mask[..., None]
            normalizer = torch.clamp(mask_expanded.sum() * mean.shape[-1], min=1.0)

            # Requested objective: MSE between sampled observation and ground truth.
            sample_sq = (sample - target_obs) ** 2
            mean_sq = (mean - target_obs) ** 2
            sample_mse = (sample_sq * mask_expanded).sum() / normalizer
            mean_mse = (mean_sq * mask_expanded).sum() / normalizer
            result["loss"] = sample_mse
            result["sample_mse"] = sample_mse
            result["mean_mse"] = mean_mse
            result["target_mask"] = mask
        return result

    def compute_tracking_error(self, pred_obs, action_pose):
        if pred_obs.ndim == 3 and action_pose.ndim == 3:
            pred_pose = pred_obs[:, :, : self.pose_action_dim]
            cmd_pose = action_pose[:, :, : self.pose_action_dim]
            return cmd_pose - pred_pose
        if pred_obs.ndim == 2 and action_pose.ndim == 2:
            pred_pose = pred_obs[:, : self.pose_action_dim]
            cmd_pose = action_pose[:, : self.pose_action_dim]
            return cmd_pose - pred_pose
        raise ValueError(
            f"Expected both tensors to be 2D or 3D, got {tuple(pred_obs.shape)} and {tuple(action_pose.shape)}."
        )
