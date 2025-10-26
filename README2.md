```markdown
## Topics in .pkl File

The following topics are included in the `.pkl` file:

- `/joint_impedance_command_controller/joint_trajectory`
- `/franka/right/obs_franka_state`
- `/franka/right/obs_franka_torque`
- `/franka_robot_state_broadcaster/measured_joint_states`
- `/franka_robot_state_broadcaster/external_joint_torques`
- `/realsense/arm/im`
- `/realsense/arm/depth`
```




## Persistent Terminal Session with tmux

To ensure your terminal session continues running even after disconnecting from SSH, follow these steps:

1. Start a new tmux session:
    ```bash
    tmux new -s factr_train
    ```

2. Activate your conda environment and run the training script:
    ```bash
    conda activate factr
    ./train_bc.sh
    ```

3. Detach from the tmux session without stopping it:
    - Press `Ctrl + B`, then `D`.

4. To reattach to the tmux session later:
    ```bash
    tmux attach -t factr_train
    ```

5. To terminate the session:
    ```bash
    tmux kill-session -t factr_train
    ```

6. Additional tmux commands:
    - List all active sessions:
      ```bash
      tmux ls
      ```
    - Reattach to a specific session:
      ```bash
      tmux attach -t <name>
      ```

