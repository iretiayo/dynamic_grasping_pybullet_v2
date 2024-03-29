#!/usr/bin/env bash

exp_name="exp_test"
timestr=${exp_name}_ur5_$(hostname)_$(date '+%Y-%m-%d_%H-%M-%S')
echo $timestr
node_id=10000

mkdir $timestr
cp run_dynamic_with_motion_parallel_ur.sh $timestr

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do

    gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        roslaunch ur5_robotiq_moveit_config demo.launch planner:=chomp; $SHELL'"
    sleep 3

    gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        python run_dynamic_with_motion.py \
            --object_name $object_name \
            --robot_config_name ur5_robotiq \
            --rendering \
            --num_trials 100 \
            --result_dir $timestr \
            --grasp_database_path assets/grasps/filtered_grasps_noise_robotiq_100_1.00 \
            --grasp_threshold 0.03 \
            --lazy_threshold  0.3 \
            --conveyor_speed 0.03 \
            --close_delay 0.5 \
            --back_off -0.05 \
            --distance_low 0.3 \
            --distance_high 0.7 \
            --pose_freq 5 \
            --record_videos \
            --max_check 3 \
            --use_box \
            --use_kf; $SHELL'"
    sleep 3
    ((node_id++))
done
