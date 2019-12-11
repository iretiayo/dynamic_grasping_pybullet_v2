#!/usr/bin/env bash

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
    python collect_good_grasps.py \
    --object_name $object_name \
    --load_folder_path assets/grasps/raw_grasps \
    --save_folder_path assets/grasps/filtered_grasps_noise \
    --min_success_rate 0.95 \
    --num_trials 50 \
    --num_grasps 5000 \
    --apply_noise \
    --back_off 0.05;
    sleep 5;
done