#!/usr/bin/env bash

timestr=$(date '+%Y-%m-%d_%H-%M-%S')
for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
    python run_static_with_motion.py \
    --object_name $object_name \
    --rendering \
    --num_trials 500 \
    --result_dir $timestr \
    --grasp_database_path assets/grasps/filtered_grasps_noise;
    sleep 5
done