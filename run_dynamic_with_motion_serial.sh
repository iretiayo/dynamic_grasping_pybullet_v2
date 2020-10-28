#!/usr/bin/env bash

exp_name="exp_name"
timestr=${exp_name}_$(hostname)_$(date '+%Y-%m-%d_%H-%M-%S')

mkdir $timestr
cp run_dynamic_with_motion_serial.sh $timestr

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
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
    --approach_prediction \
    --fix_motion_planning_time 0.14 \
    --fix_grasp_ranking_time 0.135;
    sleep 5
done
