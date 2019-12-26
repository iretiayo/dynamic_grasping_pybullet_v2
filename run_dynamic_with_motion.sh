#!/usr/bin/env bash

timestr=$(date '+%Y-%m-%d_%H-%M-%S')
for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
    python run_dynamic_with_motion.py \
    --object_name $object_name \
    --rendering \
    --num_trials 100 \
    --result_dir $timestr \
    --grasp_database_path assets/grasps/filtered_grasps_noise_100 \
    --grasp_threshold 0.03 \
    --lazy_threshold  0.3 \
    --conveyor_speed 0.01 \
    --close_delay 0.5 \
    --pose_greq 5 \
    --use_gt;
    sleep 5
done