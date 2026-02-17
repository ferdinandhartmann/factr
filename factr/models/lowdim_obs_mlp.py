import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, obs_window, action_chunk, stiffness_labels, goal_labels, target_obs=None, target_mask=None):
        features = self._build_features(obs_window, action_chunk, stiffness_labels, goal_labels)
        out = self.backbone(features).view(features.shape[0], self.pred_horizon, self.predict_obs_dim, 2)
        mean = out[..., 0]
        var = F.softplus(out[..., 1]) + self.min_var
        std = torch.sqrt(var)
        sample = mean + torch.randn_like(mean) * std

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
