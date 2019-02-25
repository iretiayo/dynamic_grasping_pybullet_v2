#!/usr/bin/env bash

iter=$1
mkdir -p log

for i in `seq 1 $iter`
do
	echo "iteration: $i"
	log_file_name="log/$(date '+%Y-%m-%d-%H-%M-%S').log"
	echo "$log_file_name"
	gnome-terminal -e "bash -ci '\
	    cd ~/dynamic_grasping_pybullet && \
	    python demo.py -v 0.03 -d 0.6 > $log_file_name;'"

	sleep 3

	gnome-terminal -e "bash -ci '\
	    roscd motion_prediction/scripts && \
	    python motion_prediction_server.py;'"

	sleep 1

	gnome-terminal -e "bash -ci '\
	    roscd pybullet_trajectory_execution/scripts && \
	    python trajectory_execution_server.py;'"

	wait=true
	while [ "$wait" = true ]
	do
		if [ -z "$(pgrep -f demo.py)" ] # if it is not running
		then
			wait=false
		else
			sleep 0.5
		fi
	done
	echo "finished"

	# make sure everything is closed
	if [ ! -z "$(pgrep -f trajectory_execution_server.py)" ]
	then
		kill -9 $(pgrep -f trajectory_execution_server.py)
	fi

	if [ ! -z "$(pgrep -f motion_prediction_server.py)" ]
	then
		kill -9 $(pgrep -f motion_prediction_server.py)
	fi
done
