#!/usr/bin/env bash
# Usage: ./collect_good_grasps.sh 3  would run in batch of 3s

object_names=(bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill)

ncpu=${1:-1}    # optional argument ncpu defaults to 1
num_objects=${#object_names[@]}
num_batches=$(((num_objects + ncpu - 1) / ncpu))

source ../../devel/setup.bash

for ((i = 0; i < num_batches; i++)); do
  echo "batch $i"
  for ((j = 0; j < ncpu; j++)); do

    idx=$((i * ncpu + j))
    object_name=${object_names[$idx]}
    echo $idx $object_name

    python collect_good_grasps.py \
      --object_name $object_name \
      --load_folder_path assets/grasps/raw_grasps \
      --save_folder_path assets/grasps/filtered_grasps_noise_100 \
      --min_success_rate 1.0 \
      --num_trials 50 \
      --num_grasps 5000 \
      --num_successful_grasps 100 \
      --apply_noise \
      --prior_folder_path assets/grasps/filtered_grasps_noise_5000_1.00 \
      --prior_success_rate 1.0 \
      --back_off 0.05 &
    sleep 5

  done
  echo "waiting"
  wait
  echo "finished waiting..."
done

#for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
#do
#    python collect_good_grasps.py \
#    --object_name $object_name \
#    --load_folder_path assets/grasps/raw_grasps \
#    --save_folder_path assets/grasps/filtered_grasps_noise_100 \
#    --min_success_rate 1.0 \
#    --num_trials 50 \
#    --num_grasps 5000 \
#    --num_successful_grasps 100 \
#    --apply_noise \
#    --prior_folder_path assets/grasps/filtered_grasps_noise_5000_1.00 \
#    --prior_success_rate 1.0 \
#    --back_off 0.05;
#    sleep 5;
#done
