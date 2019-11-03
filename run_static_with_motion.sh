#!/usr/bin/env bash

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
    python run_static_with_motion.py --object_name $object_name --rendering --num_trials 500 --result_dir result --grasp_database_path yeah
    sleep 5
done