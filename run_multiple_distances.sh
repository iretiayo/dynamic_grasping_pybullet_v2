#!/usr/bin/env bash

node_id=10000
for distance_low in 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85; do
  incremental=0.05;
  distance_high=$( bc <<<"$distance_low + $incremental" );
  ./run_single.sh \
    --node_id ${node_id} \
    --exp_name ${distance_low}_${distance_high} \
    --distance_low ${distance_low} \
    --distance_high ${distance_high};
  echo $node_id;
  echo ${distance_low};
  echo ${distance_high};
  echo '';
  let "node_id=node_id+1";
  sleep 5;
done;
