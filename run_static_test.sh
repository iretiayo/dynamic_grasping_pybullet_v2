#!/usr/bin/env bash

# usage: 
# for n in {0..2}; do ./run_static_test.sh $n; sleep 10; done

grasp_planning_type=${1:-0}     # default grasp planning type is 0 (uniform), others are 1 (sim ann) and 2 (reachability aware)

echo "grasp_planning_type: $grasp_planning_type"


if [  "$grasp_planning_type" = 0  ]
then
    gnome-terminal -e "bash -ci '\
    source ../../devel/setup.bash && \
    roslaunch grid_sample_plugin grid_sample_plugin.launch ;$SHELL'"
    sleep 3

elif [ "$grasp_planning_type" = 1 ]
then
    gnome-terminal -e "bash -ci '\
    source ../../devel/setup.bash && \
    roslaunch graspit_interface graspit_interface.launch;$SHELL'"
    sleep 3

elif [ "$grasp_planning_type" = 2 ]
then
    gnome-terminal -e "bash -ci '\
    source ../../devel/setup.bash && \
    roslaunch mico_reachability_config reachability_energy_plugin.launch;$SHELL'"
    sleep 3

fi

results_dir=results_$(date '+%Y-%m-%d-%H:%M:%S')
echo "results_dir: $results_dir"

# for object_name in bleach_cleanser # mug mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
for object_name in bleach_cleanser master_chef_can power_drill tomato_soup_can bowl mug power_drill_bkup cracker_box mustard_bottle pudding_box cube potted_meat_can sugar_box
    do
        for object_location_x in 0.3 0.4
            do
                for object_location_y in 0.0 0.2
                    do
                    source ../../devel/setup.bash && \
                        python static_grasping_analysis.py -o $object_name -x $object_location_x -y $object_location_y\
                            -g $grasp_planning_type -rd $results_dir;
                    wait
                    sleep 5
                done
        done
done