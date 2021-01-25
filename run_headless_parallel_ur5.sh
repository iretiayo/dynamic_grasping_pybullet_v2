#!/usr/bin/env bash

exp_name="exp"
timestr=${exp_name}_ur5_$(hostname)_$(date '+%Y-%m-%d_%H-%M-%S')

mkdir $timestr
cp run_headless_parallel_ur5.sh $timestr

node_id=10000

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do

    screen -d -m -S ${object_name}_moveit bash -c "source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:${node_id} && \
        roslaunch launch/ur5_robotic_moveit_ros.launch planner:=ompl;"

    sleep 3

    screen -d -m -S ${object_name}_pybullet bash -c "source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        python run_dynamic_with_motion.py \
            --object_name $object_name \
            --robot_config_name ur5_robotiq \
            --num_trials 100 \
            --result_dir $timestr \
            --grasp_database_path assets/grasps/filtered_grasps_noise_robotiq_100_1.00 \
            --baseline_experiment_path assets/results/dynamic/ur5_dynamic_medium_speed \
            --grasp_threshold 0.1 \
            --lazy_threshold  0.3 \
            --conveyor_speed 0.03 \
            --close_delay 0.5 \
            --back_off -0.075 \
            --distance_low 0.3 \
            --distance_high 0.7 \
            --pose_freq 5 \
            --max_check 3 \
            --use_box \
            --use_kf; $SHELL"
    sleep 3
    ((node_id++))
done
