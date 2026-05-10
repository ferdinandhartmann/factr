# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn.functional as F

from factr.trainers.base import BaseTrainer


def _rot6d_to_matrix(rot6):
    a1 = rot6[..., :3]
    a2 = rot6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2_orth = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_orth, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    b3 = torch.nn.functional.normalize(b3, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _smooth_trajectory(actions, smoothing_factor):
    factor = float(smoothing_factor)
    if factor <= 0.0:
        return actions
    if actions.shape[-2] < 3:
        return actions

    kernel = torch.tensor([0.25, 0.5, 0.25], device=actions.device, dtype=actions.dtype).view(1, 1, 3)
    bsz, ns, steps, dims = actions.shape
    flat = actions.permute(0, 1, 3, 2).reshape(-1, 1, steps)
    smoothed = F.conv1d(flat, kernel, padding=1)
    smoothed = smoothed.reshape(bsz, ns, dims, steps).permute(0, 1, 3, 2)
    smoothed = actions * (1.0 - factor) + smoothed * factor

    smoothed[..., 0, :] = actions[..., 0, :]
    smoothed[..., -1, :] = actions[..., -1, :]
    return smoothed


def _apply_diversity_margin(pairwise_vals, margin):
    if margin is None:
        return pairwise_vals
    margin_val = float(margin)
    if margin_val <= 0.0:
        return pairwise_vals
    return torch.relu(pairwise_vals - margin_val)


def _pairwise_endpoint_position_diversity(sampled_actions, mask, margin=None, smoothing=0.0):
    sampled_actions = _smooth_trajectory(sampled_actions, smoothing)
    endpoints = sampled_actions[:, :, -1, :3]
    endpoint_mask = mask[:, -1, :3].mean(dim=1, keepdim=True)
    diffs = endpoints.unsqueeze(2) - endpoints.unsqueeze(1)
    dist = torch.sqrt((diffs.pow(2)).sum(dim=-1) + 1e-12)
    tri_i, tri_j = torch.triu_indices(
        sampled_actions.shape[1],
        sampled_actions.shape[1],
        offset=1,
        device=sampled_actions.device,
    )
    pairwise_vals = dist[:, tri_i, tri_j] * endpoint_mask
    pairwise_vals = _apply_diversity_margin(pairwise_vals, margin)
    return pairwise_vals.mean()


def _pairwise_endpoint_orientation_diversity(sampled_actions, mask, margin=None, smoothing=0.0):
    sampled_actions = _smooth_trajectory(sampled_actions, smoothing)
    endpoint_rot6 = sampled_actions[:, :, -1, 3:9]
    endpoint_mask = mask[:, -1, 3:9].mean(dim=1, keepdim=True)
    rot = _rot6d_to_matrix(endpoint_rot6)
    rel = torch.matmul(rot.unsqueeze(2).transpose(-1, -2), rot.unsqueeze(1))
    trace_rel = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = torch.clamp((trace_rel - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    tri_i, tri_j = torch.triu_indices(
        sampled_actions.shape[1],
        sampled_actions.shape[1],
        offset=1,
        device=sampled_actions.device,
    )
    pairwise_vals = theta[:, tri_i, tri_j] * endpoint_mask
    pairwise_vals = _apply_diversity_margin(pairwise_vals, margin)
    return pairwise_vals.mean()


def _pairwise_trajectory_position_diversity(sampled_actions, mask, margin=None, smoothing=0.0, debug=False):
    # sampled_actions: [B, S, T, D]
    # mask: [B, T, D]
    sampled_actions = _smooth_trajectory(sampled_actions, smoothing)
    traj = sampled_actions[..., :3]  # [B, S, T, 3]
    traj_mask = mask[:, None, :, :3].mean(dim=-1)  # [B, 1, T]

    diffs = traj.unsqueeze(2) - traj.unsqueeze(1)  # [B, S, S, T, 3]
    dist = torch.sqrt((diffs.pow(2)).sum(dim=-1) + 1e-12)  # [B, S, S, T]

    tri_i, tri_j = torch.triu_indices(
        sampled_actions.shape[1],
        sampled_actions.shape[1],
        offset=1,
        device=sampled_actions.device,
    )

    pairwise_vals = dist[:, tri_i, tri_j, :] * traj_mask  # [B, P, T]
    margin_vals = _apply_diversity_margin(pairwise_vals, margin)
    if debug:
        with torch.no_grad():
            valid = traj_mask > 0
            valid_frac = float(valid.float().mean().item()) if valid.numel() > 0 else 0.0
            raw_min = float(pairwise_vals.min().item()) if pairwise_vals.numel() > 0 else 0.0
            raw_mean = float(pairwise_vals.mean().item()) if pairwise_vals.numel() > 0 else 0.0
            raw_max = float(pairwise_vals.max().item()) if pairwise_vals.numel() > 0 else 0.0
            loss_mean = float(margin_vals.mean().item()) if margin_vals.numel() > 0 else 0.0
            loss_pos = float((margin_vals > 0).float().mean().item()) if margin_vals.numel() > 0 else 0.0
            print(
                "[diversity] traj_pos"
                f" smoothing={float(smoothing):.3f}"
                f" margin={float(margin) if margin is not None else None}"
                f" valid_frac={valid_frac:.3f}"
                f" dist(min/mean/max)={raw_min:.4f}/{raw_mean:.4f}/{raw_max:.4f}"
                f" relu_mean={loss_mean:.4f}"
                f" relu_pos_frac={loss_pos:.3f}"
            )
    return margin_vals.mean()


class BehaviorCloning(BaseTrainer):
    def __init__(
        self,
        model,
        device_id,
        optim_builder,
        schedule_builder=None,
        train_log_freq=100,
        diversity_enable=False,
        diversity_num_samples=8,
        diversity_weight=0.02,
        diversity_weight_position=1.0,
        diversity_weight_orientation=0.5,
        diversity_warmup_steps=500,
        diversity_margin=2.0,
        diversity_smoothing=0.0,
        diversity_debug=False,
    ):
        super().__init__(
            model=model,
            device_id=device_id,
            optim_builder=optim_builder,
            schedule_builder=schedule_builder,
            train_log_freq=train_log_freq,
        )
        self.diversity_enable = bool(diversity_enable)
        self.diversity_num_samples = int(diversity_num_samples)
        self.diversity_weight = float(diversity_weight)
        self.diversity_weight_position = float(diversity_weight_position)
        self.diversity_weight_orientation = float(diversity_weight_orientation)
        self.diversity_warmup_steps = int(diversity_warmup_steps)
        self.diversity_margin = float(diversity_margin)
        self.diversity_smoothing = float(diversity_smoothing)
        self.diversity_debug = bool(diversity_debug)

    def _sample_actions_prior_with_grad(self, model, imgs, obs, labels):
        if all(
            hasattr(model, name)
            for name in (
                "_build_context_tokens",
                "_build_z_context",
                "_prior",
                "_sample_latent_batch",
                "_prepare_decoder_latent",
                "_decode_actions",
            )
        ):
            context_tokens = model._build_context_tokens(obs, class_labels=labels)
            z_context = model._build_z_context(context_tokens)
            prior_params = model._prior(z_context)
            z = model._sample_latent_batch(prior_params, sample=True, num_samples=self.diversity_num_samples)
            z = model._prepare_decoder_latent(z)
            batch_size, _, z_dim = z.shape
            context_tokens = context_tokens.repeat_interleave(self.diversity_num_samples, dim=0)
            z = z.reshape(batch_size * self.diversity_num_samples, z_dim)
            action_pred = model._decode_actions(context_tokens, z)
            return action_pred.view(batch_size, self.diversity_num_samples, model._ac_chunk, model._ac_dim)
        return model.get_actions_prior(
            imgs,
            obs,
            class_labels=labels,
            sample=True,
            num_samples=self.diversity_num_samples,
        )

    def training_step(self, batch, global_step):
        (imgs, obs), actions, mask, labels = batch
        imgs = {k: v.to(self.device_id) for k, v in imgs.items()}
        obs, actions, mask, labels = [ar.to(self.device_id) for ar in (obs, actions, mask, labels)]

        ac_flat = actions.reshape((actions.shape[0], -1))
        mask_flat = mask.reshape((mask.shape[0], -1))

        loss_dict = self.model(imgs, obs, ac_flat, mask_flat, class_labels=labels)

        def reduce_loss(t):
            return t.mean() if t.ndim > 0 else t

        total_loss = reduce_loss(loss_dict["total_loss"])

        if self.is_train:
            base_total_loss = total_loss

            if self.diversity_enable:
                model_core = self.model.module if hasattr(self.model, "module") else self.model
                sampled_actions = self._sample_actions_prior_with_grad(model_core, imgs, obs, labels)
                # endpoint_position_div = _pairwise_endpoint_position_diversity(sampled_actions, mask)
                endpoint_position_div = _pairwise_trajectory_position_diversity(
                    sampled_actions,
                    mask,
                    margin=self.diversity_margin,
                    smoothing=self.diversity_smoothing,
                    debug=self.diversity_debug,
                )
                endpoint_orientation_div = _pairwise_endpoint_orientation_diversity(
                    sampled_actions,
                    mask,
                    margin=self.diversity_margin,
                    smoothing=self.diversity_smoothing,
                )
                diversity_combined = (
                    self.diversity_weight_position * endpoint_position_div
                    + self.diversity_weight_orientation * endpoint_orientation_div
                )
                warmup_scale = min(1.0, float(global_step + 1) / float(self.diversity_warmup_steps))
                diversity_coeff = self.diversity_weight * warmup_scale
                diversity_loss_term = diversity_coeff * diversity_combined
                total_loss = total_loss - diversity_loss_term

                self.log("endpoint_position_diversity", global_step, endpoint_position_div.detach().item())
                self.log("endpoint_orientation_diversity_rad", global_step, endpoint_orientation_div.detach().item())
                self.log(
                    "endpoint_orientation_diversity_deg",
                    global_step,
                    float(endpoint_orientation_div.detach().item() * (180.0 / np.pi)),
                )
                self.log("diversity_combined", global_step, diversity_combined.detach().item())
                self.log("diversity_loss_term", global_step, diversity_loss_term.detach().item())

            self.log("total_loss", global_step, total_loss.item())
            self.log("base_total_loss", global_step, base_total_loss.item())
            self.log("lr", global_step, self.lr)

            for k, v in loss_dict.items():
                if k in ["logits", "total_loss", "logits"]:
                    continue

                if v is not None:
                    val = reduce_loss(v)
                    if hasattr(val, "item"):
                        val = val.item()

                    self.log(k, global_step, val)

            if loss_dict.get("l1_loss") is not None:
                posterior_l1 = reduce_loss(loss_dict["l1_loss"])
                if hasattr(posterior_l1, "item"):
                    posterior_l1 = posterior_l1.item()
                self.log("posterior_l1", global_step, posterior_l1)

        self.last_train_loss = total_loss.item()

        return total_loss
