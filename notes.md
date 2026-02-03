## Copy folders from other computer to this computer

### Copy one folder to the currect folder

```bash
rsync -avzP --inplace --nocompression otake@192.168.1.172:/home/otake/factr_ws/raw_data/fourgoals_1_stiff .
```
### Copy multiple folders at once to the currect folder

```bash
rsync -avP otake@192.168.1.172:/home/otake/factr_ws/raw_data/{fourgoals_1_stiff,fourgoals_1_medium,fourgoals_1_soft} .
```

## Topics in .pkl File

The following topics are included in the `.pkl` file:

- `/joint_impedance_command_controller/joint_trajectory` use for training
- `/franka/right/obs_franka_state` from follower, published again
- `/franka/right/obs_franka_torque` from follower, published again
- `/franka_robot_state_broadcaster/measured_joint_states` from follower 
- `/franka_robot_state_broadcaster/external_joint_torques` from follower 
- `/realsense/arm/im`  
- `/realsense/arm/depth`


# 🦾 FACTR Teleoperation Topics Overview

| **Topic** | **Publisher** | **Type** | **Direction** | **Description** |
|------------|---------------|-----------|----------------|------------------|
| `/franka/<side>/obs_franka_state` | `FACTRTeleopFrankaZMQ` | `sensor_msgs/JointState` | **Follower → ROS** | Mirror of the follower Franka’s current joint positions (received via ZMQ, re-published into ROS for logging/training). |
| `/franka/<side>/obs_franka_torque` | `FACTRTeleopFrankaZMQ` | `sensor_msgs/JointState` | **Follower → ROS** | External joint torques of the follower Franka, used for force feedback or contact analysis. |
| `/factr_teleop/<side>/cmd_franka_pos` | `FACTRTeleopFrankaZMQ` | `sensor_msgs/JointState` | **Leader → Follower** | Commanded joint positions from the leader arm to the follower robot. Represents desired motion. |
| `/factr_teleop/<side>/cmd_gripper_pos` | `FACTRTeleopFrankaZMQ` | `sensor_msgs/JointState` | **Leader → Follower** | Commanded gripper open/close position from the leader gripper to the follower gripper. |
| `/gripper/<side>/obs_gripper_torque` | Follower gripper driver | `sensor_msgs/JointState` | **Follower → ROS** | Measured torque (or load) on the follower’s gripper fingers, used for haptic feedback. |
| `/franka_robot_state_broadcaster/external_joint_torques` | Franka ROS driver | `sensor_msgs/JointState` | **Robot → ROS** | Native Franka topic publishing estimated external torques (used by FACTR for reference or debugging). |


## Check Folder File Sizes
To check the size of all files in the folder, sorted and displayed in a human-readable format, use the following command:

```bash
du -ah | sort -h
```

- `du -ah`: Displays the disk usage of all files and directories in human-readable format.
- `--max-depth=1`: Limits the depth to the current folder.
- `sort -h`: Sorts the output by size in human-readable format.

Check the space of the whole computer
```bash
df -h /
```

## Check Folder Disk Usage

To check the disk usage of files and directories in the current folder, sorted by size, use the following command:

```bash
du -sh * | sort -h
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


rsync -avz -e ssh otake@192.168.1.172:~/factr_ws/raw_data/box_lift_3 .

## Delete All JSON Files in the Folder

To delete all `.json` files in the current folder, use the following command:

```bash
rm *.json
```

- `rm`: Command to remove files.
- `*.json`: Matches all files with the `.json` extension in the current folder.

**Caution:** This command is irreversible. Double-check the folder contents before running it.

conda install -c conda-forge roboticstoolbox-python


when moving folder and the factr library doesnt work anymore:
python -m pip uninstall -y factr
python -m pip install -e .