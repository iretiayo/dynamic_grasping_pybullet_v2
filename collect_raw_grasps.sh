#!/usr/bin/env bash

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
    python collect_raw_grasps.py --object_name $object_name --robot_name MicoGripper --num_grasps 5000 --max_steps 40000 --grasp_folder_path raw_grasps
    sleep 5
done