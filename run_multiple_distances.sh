#!/usr/bin/env bash

node_id=10000
# safe distance for ur5 is 0.25 <= 0.90
for distance_low in 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75; do
  incremental=0.05;
  distance_high=$( bc <<<"$distance_low + $incremental" );
  ./run_single.sh \
    --node_id ${node_id} \
    --exp_name ur5_${distance_low}_${distance_high} \
    --motion_mode dynamic_linear \
    --distance_low ${distance_low} \
    --distance_high ${distance_high} \
    --use_reachability true \
    --use_motion_aware false \
    --conveyor_speed 0.05 ;
    --baseline_experiment_path None;
  echo $node_id;
  echo ${distance_low};
  echo ${distance_high};
  echo '';
  let "node_id=node_id+1";
  sleep 5;
done;
