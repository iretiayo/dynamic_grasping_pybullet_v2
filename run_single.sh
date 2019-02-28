#!/usr/bin/env bash

iter=$1
object_name=$2
conveyor_velocity=$3
conveyor_distance=$4

echo "iter $iter conveyor_velocity $conveyor_velocity conveyor_distance $conveyor_distance"
mkdir -p log

for i in `seq 1 $iter`
do
	echo "iteration: $i"
	log_file_name="log/$(date '+%Y-%m-%d-%H-%M-%S').log"
	echo "$log_file_name"
	gnome-terminal -e "bash -ci '\
	    source ../../devel/setup.bash && \
	    python demo.py -o $object_name -v $conveyor_velocity -d $conveyor_distance > $log_file_name;'"
    demo_pid=$!
	sleep 3

	gnome-terminal -e "bash -ci '\
	    source ../../devel/setup.bash && \
	    roscd motion_prediction/scripts && \
	    python motion_prediction_server.py;'"
    kalman_pid=$!
	sleep 1

	gnome-terminal -e "bash -ci '\
	    source ../../devel/setup.bash && \
	    roscd pybullet_trajectory_execution/scripts && \
	    python trajectory_execution_server.py;'"
    execution_pid=$!

	while kill -0 "$demo_pid" >/dev/null 2>&1; do
        echo "demo.py PROCESS IS RUNNING"
    done
    kill -9 $kalman_pid
    kill -9 $execution_pid
done
