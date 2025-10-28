```markdown
## Topics in .pkl File

The following topics are included in the `.pkl` file:

- `/joint_impedance_command_controller/joint_trajectory` use for training
- `/franka/right/obs_franka_state` from follower, published again
- `/franka/right/obs_franka_torque` from follower, published again
- `/franka_robot_state_broadcaster/measured_joint_states` from follower 
- `/franka_robot_state_broadcaster/external_joint_torques` from follower 
- `/realsense/arm/im`  
- `/realsense/arm/depth`
```

# 🦾 FACTR Teleoperation Topics Overview

| **Topic** | **Publisher** | **Type** | **Direction** | **Description** |
|------------|---------------|-----------|----------------|------------------|
| `/franka/<side>/obs_franka_state` | `FACTRTeleopFrankaZMQ` | `sensor_msgs/JointState` | **Follower → ROS** | Mirror of the follower Franka’s current joint positions (received via ZMQ, re-published into ROS for logging/training). |
| `/franka/<side>/obs_franka_torque` | `FACTRTeleopFrankaZMQ` | `sensor_msgs/JointState` | **Follower → ROS** | External joint torques of the follower Franka, used for force feedback or contact analysis. |
| `/factr_teleop/<side>/cmd_franka_pos` | `FACTRTeleopFrankaZMQ` | `sensor_msgs/JointState` | **Leader → Follower** | Commanded joint positions from the leader arm to the follower robot. Represents desired motion. |
| `/factr_teleop/<side>/cmd_gripper_pos` | `FACTRTeleopFrankaZMQ` | `sensor_msgs/JointState` | **Leader → Follower** | Commanded gripper open/close position from the leader gripper to the follower gripper. |
| `/gripper/<side>/obs_gripper_torque` | Follower gripper driver | `sensor_msgs/JointState` | **Follower → ROS** | Measured torque (or load) on the follower’s gripper fingers, used for haptic feedback. |
| `/franka_robot_state_broadcaster/external_joint_torques` | Franka ROS driver | `sensor_msgs/JointState` | **Robot → ROS** | Native Franka topic publishing estimated external torques (used by FACTR for reference or debugging). |



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

