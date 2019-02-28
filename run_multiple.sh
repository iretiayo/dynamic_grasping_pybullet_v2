#!/usr/bin/env bash

iter=1
#for object_name in cube bleach_cleanser bowl cracker_box master_chef_can mug mustard_bottle potted_meat_can power_drill pudding_box sugar_box tomato_soup_can
for object_name in cube
do
    for conveyor_distance in 0.6
    do
        for conveyor_velocity in 0.01
        do
            echo "./run_single.sh $iter $object_name $conveyor_velocity $conveyor_distance"
            ./run_single.sh $iter $object_name $conveyor_velocity $conveyor_distance
        done
    done
done