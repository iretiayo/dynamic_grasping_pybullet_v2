#!/usr/bin/env bash

node_id=10000
# safe distance for ur5 is 0.25 <= 0.90
for alpha in 0.2 0.4 0.5 0.6 0.8; do
  ./run_single.sh \
    --node_id ${node_id} \
    --exp_name alpha_${alpha} \
    --record_video false \
    --rendering false;
  echo node $node_id;
  echo alpha ${alpha};
  echo '';
  let "node_id=node_id+1";
  sleep 5;
done;
