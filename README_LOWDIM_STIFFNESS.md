# Lowdim Stiffness Pipeline (Quick Run)

This setup uses:
- **Obs**: 27D state + `stiffness_label` class (`1/2/3`)
- **Action target**: commanded EE pose only (`9D = [x,y,z] + 6D rotation representation`)
- **No extra normalization in FACTR training** (buffer is assumed pre-normalized)

## 1) Environment
From `factr/`:

```bash
conda activate factr
python -m pip install -e .
```

## 2) Regenerate processed buffer (with stiffness label in obs)
From `factr/`:

```bash
python process_data/process_data.py
```

Example output:
- `process_data/processed_data/fourgoals_1_act/buf.pkl`
- `process_data/processed_data/fourgoals_1_act/rollout_config.yaml`

## 3) Inspect generated buffer
From `factr/`:

```bash
python scripts/inspect_new_buffer.py \
  --buffer-path process_data/processed_data/fourgoals_1_act/buf.pkl
```

Check that:
- `obs` contains `state`, `goals`, `stiffness_label`
- action dim is `9` for this new dataset
- stiffness label counts are shown from `obs['stiffness_label']`
- legacy action-signature section says no signatures (expected for pose-only action)

## 4) Train lowdim stiffness-conditioned model
From `factr/`:

```bash
python factr/train_bc_policy.py --config-name train_bc_lowdim
```

Optional quick smoke run:

```bash
python factr/train_bc_policy.py --config-name train_bc_lowdim \
  max_iterations=1000 eval_freq=100 save_freq=500
```

## 5) W&B logs you should see
During eval, these are logged:
- `eval/task_loss`
- `eval/action_l2`
- `eval/action_lsig`
- `eval/pose_dim*_l2` (per EE-pose dimension L2)
- `eval/plot_error_summary` (Matplotlib image: chunk-step MSE + per-dim MSE)
- `eval/plot_example_gt_vs_pred` (Matplotlib image: ground truth vs predicted pose trajectory)

## 6) Prior sampling rollout (multi-trajectory)
From `factr/`:

```bash
python scripts/test_rollout_lowdim_prior.py \
  --model-name <checkpoint_folder_name> \
  --checkpoint latest \
  --buffer-path process_data/processed_data/fourgoals_1_act/buf.pkl \
  --episode-index 0 \
  --num-samples 10 \
  --stiffness-label 1
```

Outputs are saved to:
- `scripts/rollout_lowdim_output/*.npz`

## 7) Evaluate one episode with plots (10 sampled trajectories per step)
From `factr/`:

```bash
python scripts/eval_one_episode_lowdim.py \
  --model-name <checkpoint_folder_name> \
  --checkpoint latest \
  --buffer-path process_data/processed_data/fourgoals_1_act/buf.pkl \
  --episode-index 0 \
  --num-samples 10 \
  --stiffness-label 0
```

This saves:
- `*_pose_traj.png` (true vs sampled trajectories for all 9 EE-pose dims)
- `*_metrics.png` (sample MSE, per-dim MSE, MAE over time)
- `*.npz` (raw sampled arrays + ground truth)
