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
iter=$1
object_name=$2
conveyor_velocity=$3
conveyor_distance=$4

echo "iter $iter object_name $object_name conveyor_velocity $conveyor_velocity conveyor_distance $conveyor_distance"
mkdir -p log

for i in `seq 1 $iter`
do
	echo "iteration: $i"
	log_file_name="log/$(date '+%Y-%m-%d-%H-%M-%S').log"
	echo "$log_file_name"

    if [  "$online_planning" = true  ]
    then
        gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        roslaunch mico_reachability_config reachability_energy_plugin.launch'"
        sleep 3
    fi


	gnome-terminal -e "bash -ci '\
	    source ../../devel/setup.bash && \
	    python demo.py -o $object_name -v $conveyor_velocity -d $conveyor_distance > $log_file_name;'"
	sleep 3

	gnome-terminal -e "bash -ci '\
	    source ../../devel/setup.bash && \
	    roscd motion_prediction/scripts && \
	    python motion_prediction_server.py;'"
	sleep 1

	gnome-terminal -e "bash -ci '\
	    source ../../devel/setup.bash && \
	    roscd pybullet_trajectory_execution/scripts && \
	    python trajectory_execution_server.py;'"
	sleep 1

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
	if [ ! -z "$(pgrep -f reachability_energy_plugin)" ]
	then
		kill -9 $(pgrep -f reachability_energy_plugin)
	fi

	if [ ! -z "$(pgrep -f trajectory_execution_server.py)" ]
	then
		kill -9 $(pgrep -f trajectory_execution_server.py)
	fi

	if [ ! -z "$(pgrep -f motion_prediction_server.py)" ]
	then
		kill -9 $(pgrep -f motion_prediction_server.py)
	fi
done
