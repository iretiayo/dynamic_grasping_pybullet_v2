#!/usr/bin/env bash

iter=1
online_planning=true
#bowl, cracker_box, power_drill  blocked
#for object_name in cube bleach_cleanser bowl cracker_box master_chef_can mug mustard_bottle potted_meat_can power_drill pudding_box sugar_box tomato_soup_can
#for object_name in sugar_box tomato_soup_can master_chef_can mug mustard_bottle potted_meat_can pudding_box cube
for object_name in sugar_box
do
    for conveyor_distance in 0.4
    do
        for conveyor_velocity in 0.01
        do

            if [  "$online_planning" = true  ] # if it is not running
            then
                echo "./run_single.sh -o $iter $object_name $conveyor_velocity $conveyor_distance"
                ./run_single.sh -o $iter $object_name $conveyor_velocity $conveyor_distance
            else
                echo "./run_single.sh $iter $object_name $conveyor_velocity $conveyor_distance"
                ./run_single.sh $iter $object_name $conveyor_velocity $conveyor_distance
            fi

        done
    done
done