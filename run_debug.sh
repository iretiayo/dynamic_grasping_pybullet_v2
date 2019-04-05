#!/usr/bin/env bash

# this option must be placed before mass arguments (operands)
while getopts ":o" opt; do
  case $opt in
    o)
        echo "online grasp planning"
        online_planning=true
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit -0
        ;;
    esac
done

shift $((OPTIND-1))
object_name=$1
conveyor_velocity=$2
conveyor_distance=$3

echo "object_name $object_name conveyor_velocity $conveyor_velocity conveyor_distance $conveyor_distance"
mkdir -p log

if [  "$online_planning" = true  ]
then
    gnome-terminal -e "bash -ci '\
    source ../../devel/setup.bash && \
    roslaunch mico_reachability_config reachability_energy_plugin.launch;$SHELL'"
    sleep 3
fi

gnome-terminal -e "bash -ci '\
    source ../../devel/setup.bash && \
    python demo.py -o $object_name -v $conveyor_velocity -d $conveyor_distance;$SHELL'"
sleep 3

gnome-terminal -e "bash -ci '\
    source ../../devel/setup.bash && \
    roscd motion_prediction/scripts && \
    python motion_prediction_server.py;$SHELL'"
sleep 1

gnome-terminal -e "bash -ci '\
    source ../../devel/setup.bash && \
    roscd pybullet_trajectory_execution/scripts && \
    python trajectory_execution_server.py;$SHELL'"
sleep 1
