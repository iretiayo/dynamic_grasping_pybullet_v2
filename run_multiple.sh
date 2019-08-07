#!/usr/bin/env bash

iter=10
online_planning=true
### all
#for object_name in cube bleach_cleanser bowl cracker_box master_chef_can mug mustard_bottle potted_meat_can power_drill pudding_box sugar_box tomato_soup_can
### bowl, cracker_box blocked, pudding box weird starting pose
### worshop objects set
for object_name in bleach_cleanser mug mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
#for object_name in bleach_cleanser
do
    for conveyor_distance in 0.3 0.4 0.5
    do
        for conveyor_velocity in 0.01 0.03 0.05
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