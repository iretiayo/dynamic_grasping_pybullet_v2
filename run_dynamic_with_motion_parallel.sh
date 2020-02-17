#!/usr/bin/env bash

timestr=$(date '+%Y-%m-%d_%H-%M-%S')
node_id=10000

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do

    gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        roslaunch m1n6s200_moveit_config m1n6s200_virtual_robot_demo.launch planner:=chomp; $SHELL'"
    sleep 3

    gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        python run_dynamic_with_motion.py \
            --object_name $object_name \
            --robot_config_name mico \
            --rendering \
            --num_trials 100 \
            --result_dir $timestr \
            --grasp_database_path assets/grasps/filtered_grasps_noise_100 \
            --baseline_experiment_path assets/results/dynamic/linear_motion/2020-01-10_18-33-49 \
            --grasp_threshold 0.03 \
            --lazy_threshold  0.3 \
            --conveyor_speed 0.03 \
            --close_delay 0.5 \
            --back_off 0.05 \
            --distance_low 0.15 \
            --distance_high 0.4 \
            --pose_freq 5 \
            --record_videos \
            --max_check 3 \
            --use_box \
            --use_kf \
            --approach_prediction; $SHELL'"
    sleep 3
    ((node_id++))
done

cp run_dynamic_with_motion_parallel.sh $timestr
