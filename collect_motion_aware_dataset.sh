#!/usr/bin/env bash

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
    python collect_motion_aware_dataset.py \
    --object_name $object_name \
    --grasp_database_path assets/grasps/filtered_grasps_noise_100 \
    --save_folder_path motion_aware_dataset \
    --num_trials_per_grasp 1000 \
    --disable_gui;
    sleep 5;
done