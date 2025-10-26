
Topics in .pkl file

['/joint_impedance_command_controller/joint_trajectory',
 '/franka/right/obs_franka_state',
 '/franka/right/obs_franka_torque',
 '/franka_robot_state_broadcaster/measured_joint_states',
 '/franka_robot_state_broadcaster/external_joint_torques',
 '/realsense/arm/im',
 '/realsense/arm/depth']


tmux new -s factr_train
conda activate factr
./train_bc.sh
Ctrl + B  then  D
tmux attach -t factr_train
exit OR tmux kill-session -t factr_train
tmux ls   # list all active sessions
tmux attach -t <name>   # reattach to one
