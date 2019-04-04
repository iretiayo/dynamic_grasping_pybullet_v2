#!/usr/bin/env bash

object_name=$1
conveyor_velocity=$2
conveyor_distance=$3

echo "object_name $object_name conveyor_velocity $conveyor_velocity conveyor_distance $conveyor_distance"
mkdir -p log

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
