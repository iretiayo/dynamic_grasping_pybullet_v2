#!/usr/bin/env bash

node_id=10000
for circular_distance_low in 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85; do
  incremental=0.05
  circular_distance_high=$( bc <<<"$circular_distance_low + $incremental" )
  ./run_single.sh \
    --node_id ${node_id} \
    --exp_name ${circular_distance_low}_${circular_distance_high}
    --circular_distance_low ${circular_distance_low} \
    --circular_distance_high ${circular_distance_high}
  echo $node_id
  echo ${circular_distance_low}
  echo ${circular_distance_high}
  echo ''
  let "node_id=node_id+1"
  sleep 5
done
