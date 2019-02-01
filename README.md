## MoveIt! vs pybullet
To indicate that a joint is continuous/circular
- pybullet needs the lower limit to be larger than upper limit in the urdf. Joint type is irrelavant.
- MoveIt! needs the joint type to be `continuous`. Joint limits are irrelavant.

When planning using MoveIt!, it first converts start joint values and goal joint values of circular joints 
to be within [-pi, pi] (no matter what) and then return a plan. This plan is then adapted by
the contoller to work the current joint values.
 
## Useful Commands
xacro
```
rosrun xacro xacro --inorder -o model.urdf kinova_description/urdf/m1n6s200_standalone.xacro
```

Launch a virtual robot, MoveIt and RViz
```
roslaunch m1n6s200_moveit_config m1n6s200_virtual_robot_demo.launch
```


