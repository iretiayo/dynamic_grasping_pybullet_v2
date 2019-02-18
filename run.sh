#!/usr/bin/env bash

gnome-terminal -e "bash -ci '\
    cd ~/dynamic_grasping_pybullet && \
    python demo.py;'"

sleep 3

gnome-terminal -e "bash -ci '\
    roscd pybullet_trajectory_execution/scripts && \
    python trajectory_execution_server.py;'"

sleep 3

gnome-terminal -e "bash -ci '\
    roscd motion_prediction/scripts && \
    python motion_prediction_server.py;'"

sleep 3

gnome-terminal -e "bash -ci '\
    cd ~/dynamic_grasping_pybullet && \
    python move_conveyor.py;'"

sleep 3
