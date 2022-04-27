---
layout: default
title: {{ site.name }}
---

This is a simple PID controller written in Python3 for controlling differential-drive vehicles, such as [Heron Unmanned Surface Vehicle (USV)](https://www.clearpathrobotics.com/assets/guides/melodic/heron/index.html), [Jackal Unmanned Ground Vehicle (UGV)](http://www.clearpathrobotics.com/assets/guides/noetic/jackal/), and [Husky UGV](https://www.clearpathrobotics.com/assets/guides/kinetic/husky/index.html) to a desired position/waypoint. We will use the Jackal simulator as a running example.

Heron USV | Jackal UGV | Husky UGV
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://www.clearpathrobotics.com/assets/guides/melodic/heron/_images/heron_banner.jpg" width="300" height="100"/> |  <img src="http://www.clearpathrobotics.com/assets/guides/noetic/jackal/_images/jackal_banner.png" width="300" height="100"/> | <img src="https://www.clearpathrobotics.com/assets/guides/kinetic/husky/_images/TJM_5949_00001.jpg" width="300" height="100"/>

---

# Table of contents

* [Jackal Simulator](#jackal)
* [Tracking PID Controller](#pid)
* [(Optional) Configuration of RViz](#rviz)

---

<a name="jackal"></a>

# Jackal Simulation

To install Jackal simulator, make sure you have a working ROS installation on your Ubuntu desktop. We use Ubuntu 18.04 which corresponds to ROS `melodic`. Change the ROS distribution accordingly based on your OS version, and install the following Jackal simulator packages:

```bash
sudo apt-get install ros-melodic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-navigation
```

Make sure your `~/.bashrc` has the following settings and remember to `source ~/.bashrc` after modifying it:

```bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
```

To launch simulated Jackal in an empty world, run the following command:

```bash
# Terminal #1
roslaunch jackal_gazebo empty_world.launch
```

We should see the simulated vehicle on an open ground.

![gazebo](./assets/gazebo.png)


<a name="pid"></a>

# Tracking PID Controller

ROS Melodic does not support Python3 natively but our code is written in Python3. To bypass this problem, we can install ROS Noetic in a conda environment following the instructions given by [RoboStack](https://github.com/RoboStack/ros-noetic). If your ROS distribution supports Python3 or you have an existing conda environment, feel free to skip this step. Assuming that you have a ROS noetic conda environment called `robostackenv`, clone and build the `tracking_pid` package in a `catkin` workspace.

```bash
# Terminal #2
conda activate robostackenv
mkdir -p ~/robostack_ws/src
cd ~/robostack_ws/src
git clone https://github.com/Weizhe-Chen/tracking_pid.git
cd ..
catkin build
. devel/setup.bash  # `.` is equivalent to `source`
rosrun tracking_pid tracking_pid_node.py
```

If everything goes well, we shall see the following terminal output, showing the robot's pose.

![terminal_state](./assets/console_state.png)

Open another terminal (terminal #3), run `rviz` &rarr; select `File` &rarr; `Open Config` &rarr; find `jackal.rviz` in `tracking_pid/rviz/`.

![rviz](./assets/rviz.png)

Now we can send a goal using the `2D Nav Goal` tool.

![rviz](./assets/demo.gif)

We can adjust the controller parameters on-the-fly using `rqt_reconfigure`:

```bash
rosrun rqt_reconfigure rqt_reconfigure
```

![reconfigure](/assets/rqt_reconfigure.png)

<a name="rviz"></a>
# (Optional) Configuration of RViz

In case opening `jackal.rviz` does not work, we can configure RViz manually via the following steps. Open a new terminal and run `rviz`:

```bash
rviz
```

![rviz_0](./assets/rviz_0.png)

Change the `Fixed Frame` to `odom`.

![rviz_1](./assets/rviz_1.png)

Add a `RobotModel`.

![rviz_2](./assets/rviz_2.png)

Add marker for visualizing the goal/waypoint.

![rviz_3](./assets/rviz_3.png)

Change the marker topic to `/waypoint_marker`

![rviz_4](./assets/rviz_4.png)
