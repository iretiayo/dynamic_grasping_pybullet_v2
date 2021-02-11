#!/usr/bin/env bash

# this script iterates through the list of objects in serial

node_id=${node_id:-10000}
exp_name=${exp_name:-run}
motion_mode=${motion_mode:-dynamic_circular}
robot_config_name=${robot_config_name:-ur5_robotiq}
num_trials=${num_trials:-100}
grasp_database_path=${grasp_database_path:-assets/grasps/filtered_grasps_noise_robotiq_100_1.00}
baseline_experiment_path=${baseline_experiment_path:-ur5_dynamic_medium_speed_z_motion}
grasp_threshold=${grasp_threshold:-0.1}
lazy_threshold=${lazy_threshold:-30.3}
conveyor_speed=${conveyor_speed:-0.05}
close_delay=${close_delay:-0.5}
back_off=${back_off:-0.075}
distance_low=${distance_low:-0.15}
distance_high=${distance_high:-0.20}
circular_distance_low=${circular_distance_low:-0.15}
circular_distance_high=${circular_distance_high:-0.20}
pose_freq=${pose_freq:-5}
use_previous_jv=${use_previous_jv:-true}
use_seed_trajectory=${use_seed_trajectory:-true}
max_check=${max_check:-3}
use_box=${use_box:-true}
use_kf=${use_kf:-true}
fix_motion_planning_time=${fix_motion_planning_time:-0.14}
disable_reachability=${disable_reachability:-false}
fix_grasp_ranking_time=${fix_grasp_ranking_time:-0.135}

# assign the keyword argument values
while [[ $# -gt 0 ]]; do
  if [[ $1 == *"--"* ]]; then
    param="${1/--/}"
    declare $param="$2"
  fi
shift
done

timestr=${exp_name}_ur5_$(hostname)_$(date '+%Y-%m-%d_%H-%M-%S')
mkdir $timestr
cp run_headless_serial_ur5.sh $timestr

screen -d -m -S ${timestr}_moveit bash -c "source ../../devel/setup.bash && \
    export ROS_MASTER_URI=http://localhost:${node_id} && \
    roslaunch launch/ur5_robotic_moveit_ros.launch planner:=ompl;"

sleep 3

source ../../devel/setup.bash
export ROS_MASTER_URI=http://localhost:$node_id

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill; do
  python run_dynamic_with_motion.py \
    --motion_mode ${motion_mode} \
    --object_name ${object_name} \
    --robot_config_name ${robot_config_name} \
    --num_trials ${num_trials} \
    --result_dir $timestr \
    --grasp_database_path ${grasp_database_path} \
    --baseline_experiment_path ${baseline_experiment_path} \
    --grasp_threshold ${grasp_threshold} \
    --lazy_threshold ${lazy_threshold} \
    --conveyor_speed ${conveyor_speed} \
    --close_delay ${close_delay} \
    --back_off ${back_off} \
    --distance_low ${distance_low} \
    --distance_high ${distance_high} \
    --circular_distance_low ${circular_distance_low} \
    --circular_distance_high ${circular_distance_high} \
    --pose_freq ${pose_freq} \
    --use_previous_jv ${use_previous_jv} \
    --use_seed_trajectory ${use_seed_trajectory} \
    --max_check ${max_check} \
    --use_box ${use_box} \
    --use_kf ${use_kf} \
    --fix_motion_planning_time ${fix_motion_planning_time} \
    --disable_reachability ${disable_reachability}\
    --fix_grasp_ranking_time ${fix_grasp_ranking_time};
  sleep 5
done