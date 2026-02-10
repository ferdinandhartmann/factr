
#!/bin/bash

CUDA_DEVICE_ID=0,1,2

task_config=single_franka
buffer_path=$(pwd)/process_data/processed_data/1217_mix/buf.pkl
feature_path=$(pwd)/visual_features/vit_base/SOUP_1M_DH.pth
wandb_entity=a-otake2415-keio-university-global-page-org

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID 

python -m factr.train_bc_policy exp_name=baseline_la7_ac100_ agent.features.restore_path=$feature_path buffer_path=$buffer_path task=$task_config wandb.entity=$wandb_entity ac_chunk=100 curriculum.start_scale=7 curriculum.space=latent agent.token_dim=512 +agent.factr_baseline=True

