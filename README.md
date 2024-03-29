## TODO 
- Tier 1
    - [ ] Real robot calibration
    - [ ] Pose evaluation
    - [ ] Grasp static object
- Tier 2
    - [ ] Motion prediction
- Tier 3
    - [ ] Dynamic grasping
- Tier 4
    - [ ] Simulation results
- Tier 5
    - [ ] Writing

## Dependencies
[git lfs](https://git-lfs.github.com/) needed to download assets that contains large files, otherwise the asset files would be broken.


## Grasp Database Generation
Generate grasps
```
python grasp_evaluation_eef_only.py --object_name bleach_cleanser
```


## MoveIt! vs pybullet
To indicate that a joint is continuous/circular
- pybullet needs the lower limit to be larger than upper limit in the urdf. Joint type is irrelavant.
- MoveIt! needs the joint type to be `continuous`. Joint limits are irrelavant.

When planning using MoveIt!, it first converts start joint values and goal joint values of circular joints 
to be within [-pi, pi] (no matter what) and then return a plan. This plan is then adapted by
the contoller to work with the current joint values. This means if you want a particular joint 
to rotate 1000 cycles, you can only do it gradually because set the target to be 1000 cycles 
will have the same effect as 1 cycle.

## Grasp Planning
MoveIt! has problems planning a trajectory to the grasp pose returned by GraspIt! because it considers a grasp 
pose very close to a target as collision (even though there is actually some room). So we back off the grasp 
to get a pre-grasp. Assuming there is no obstacles between grasp and pre-grasp, we then need to check two things:
- the pre-grasp is reachable with the target
- the grasp is reachable without the target

## Modification on URDF
- Modify joint limits
- Disable collision model on `m1n6s200_end_effector`
 
## Useful Commands
xacro
```
rosrun xacro xacro --inorder -o model.urdf kinova_description/urdf/m1n6s200_standalone.xacro
```

Launch a virtual robot, MoveIt and RViz
```
roslaunch m1n6s200_moveit_config m1n6s200_virtual_robot_demo.launch
```

Launch GraspIt! with reachability plug-in
```
roslaunch mico_reachability_config reachability_energy_plugin.launch
```
We need two packages: `mico_reachability_config` and `reachability_energy_plugin`


## On Real Robot
To access the arm via usb copy the udev rule file
```
sudo cp kinova_driver/udev/10-kinova-arm.rules /etc/udev/rules.d/
```
Launch the essential drivers and configurations for kinova robots
```
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=m1n6s200
```
Launch MoveIt! and RViz
```
roslaunch m1n6s200_moveit_config m1n6s200_demo.launch
```
