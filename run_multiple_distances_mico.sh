#!/usr/bin/env bash

node_id=10000
# safe distance for ur5 is 0.25 <= 0.90
for distance_low in 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
  incremental=0.05;
  distance_high=$( bc <<<"$distance_low + $incremental" );
  ./run_single_mico.sh --node_id ${node_id} \
    --exp_name ur5_${distance_low}_${distance_high} \
    --motion_mode dynamic_linear \
    --use_reachability true \
    --use_motion_aware false \
    --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
    --conveyor_speed 0.03;
  echo $node_id;
  echo ${distance_low};
  echo ${distance_high};
  echo '';
  let "node_id=node_id+1";
  sleep 5;
done;
