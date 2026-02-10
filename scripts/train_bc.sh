
#!/bin/bash

CUDA_DEVICE_ID=0

task_config=single_franka
buffer_path=/home/ferdinand/activeinference/factr/process_data/training_data/fourgoals_1_norm2/buf.pkl
feature_path=/home/ferdinand/activeinference/factr/scripts/visual_features/vit_base/SOUP_1M_DH.pth
wandb_entity=ferdinand-hartmann-keio-university-org

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID 

python -m factr.train_bc_policy exp_name=fourgoals_1_norm2_1 agent.features.restore_path=$feature_path buffer_path=$buffer_path task=$task_config wandb.entity=$wandb_entity

