
#!/bin/bash

# CUDA_DEVICE_ID=0

# task_config=single_franka
# buffer_path=/home/ferdinand/activeinference/factr/process_data/training_data/fourgoals_1_act/buf.pkl
# feature_path=/home/ferdinand/activeinference/factr/scripts/visual_features/vit_base/SOUP_1M_DH.pth
# wandb_entity=ferdinand-hartmann-keio-university-org

CUDA_VISIBLE_DEVICES=0 

python -m factr.train_bc_policy

