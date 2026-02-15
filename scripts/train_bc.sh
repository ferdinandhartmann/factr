
#!/bin/bash
set -euo pipefail

CUDA_DEVICE_ID="${CUDA_DEVICE_ID:-0}"
task_config="${TASK_CONFIG:-single_franka_lowdim}"
buffer_path="${BUFFER_PATH:-/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_1_act/buf_train.pkl}"
test_buffer_path="${TEST_BUFFER_PATH:-/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_1_act/buf_test.pkl}"
wandb_entity="${WANDB_ENTITY:-ferdinand-hartmann-keio-university}"
config_name="${CONFIG_NAME:-train_bc_lowdim}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_ID" python -m factr.train_bc_policy \
  --config-name "${config_name}" \
  task="${task_config}" \
  buffer_path="${buffer_path}" \
  test_buffer_path="${test_buffer_path}" \
  wandb.entity="${wandb_entity}" \
  "$@"
