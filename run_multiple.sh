#!/usr/bin/env bash

# initial node id
node_id=10000

### Motion-aware
# linear 5 cm/s
echo node id, ${node_id}
./run_single.sh --node_id ${node_id} \
  --exp_name ma_linear \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/ur5_robotiq/linear_obstacles \
  --conveyor_speed 0.05;
let "node_id=node_id+1";
sleep 5;

# linear 5 cm/s, with obstacles
echo node id, ${node_id}
./run_single.sh --node_id ${node_id} \
  --exp_name ma_linear_ob \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/ur5_robotiq/linear_obstacles \
  --conveyor_speed 0.05;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with slab
echo node id, ${node_id}
./run_single.sh --node_id ${node_id} \
  --exp_name ma_linear_slab \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/ur5_robotiq/linear_obstacles \
  --conveyor_speed 0.03 \
  --add_top_shelf true;
let "node_id=node_id+1";
sleep 5;

# linear 5 cm/s, with z motion
echo node id, ${node_id}
./run_single.sh --node_id ${node_id} \
  --exp_name ma_linear_z \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/ur5_robotiq/variable_z \
  --conveyor_speed 0.05;
let "node_id=node_id+1";
sleep 5;

# linear 5 cm/s, vary speed
echo node id, ${node_id}
./run_single.sh --node_id ${node_id} \
  --exp_name ma_linear_vary \
  --motion_mode dynamic_linear_vary_speed \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/ur5_robotiq/linear_obstacles \
  --conveyor_speed 0.05;
let "node_id=node_id+1";
sleep 5;

# circular 3 cm/s
echo node id, ${node_id}
./run_single.sh --node_id ${node_id} \
  --exp_name ma_circular \
  --motion_mode dynamic_circular \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/ur5_robotiq/dynamic_circular \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# sinusoidal 1 cm/s
echo node id, ${node_id}
./run_single.sh --node_id ${node_id} \
  --exp_name ma_sinusoid \
  --motion_mode dynamic_sinusoid \
  --use_reachability false \
  --use_motion_aware true \
  --use_gt true \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/ur5_robotiq/linear_obstacles \
  --conveyor_speed 0.01;
let "node_id=node_id+1";
sleep 5;
