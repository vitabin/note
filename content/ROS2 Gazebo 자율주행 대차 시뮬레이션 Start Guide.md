---
title: ROS2 Gazebo 자율주행 대차 시뮬레이션 Start Guide
category:
  - Robotics
  - Simulation
  - AutonomousDriving
tags:
  - ROS2
  - Gazebo
  - Docker
  - LiDAR
  - URDF
  - SLAM
  - Navigation
  - AGV
---

---

# ROS2 Gazebo 자율주행 대차 시뮬레이션 Start Guide

본 문서는 `osrf/ros:humble-desktop-full` Docker 환경에서 ROS2 Humble과 Gazebo Classic을 사용하여 자율주행 대차(AGV) 시뮬레이션 실습 및 연구 환경을 구성하는 과정을 정리합니다.

현재 구성은 대차 모델 표시, `/cmd_vel` 기반 주행, LiDAR 센서 추가, `/scan` 토픽 확인, 이후 SLAM 및 Navigation2 연동을 목표로 합니다.

---

## 1. 현재 환경

| 항목 | 내용 |
| :--- | :--- |
| Docker Image | `osrf/ros:humble-desktop-full` |
| ROS2 Distro | Humble |
| Simulator | Gazebo Classic |
| Build Tool | colcon |
| Model Format | URDF / Xacro |

> 이미지 이름으로 `osrf/ros:bumle-desktop-full`을 사용했다면 오타일 가능성이 높습니다. 일반적으로는 `osrf/ros:humble-desktop-full`입니다.

---

## 2. 실습 진행 순서

1. ROS2 workspace 생성
2. AGV 시뮬레이션 패키지 생성
3. URDF/Xacro 기반 대차 모델 작성
4. Gazebo에 로봇 spawn
5. `gazebo_ros_diff_drive` 플러그인으로 `/cmd_vel` 주행
6. LiDAR 링크와 Gazebo ray sensor 추가
7. `/scan`, `/odom`, `/tf` 토픽 확인
8. RViz에서 RobotModel, TF, LaserScan 확인
9. SLAM Toolbox로 지도 작성
10. Navigation2로 목표 지점 주행

---

## 3. Workspace 구성

### 3.1 Workspace 생성 및 빌드

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 3.2 패키지 생성

```bash
cd ~/ros2_ws/src
ros2 pkg create agv_sim --build-type ament_cmake
```

추천 폴더 구조는 다음과 같습니다.

```text
agv_sim/
  launch/
  urdf/
  worlds/
  config/
  CMakeLists.txt
  package.xml
```

`CMakeLists.txt`에는 launch, urdf, worlds, config 폴더가 install 되도록 추가합니다.

```cmake
install(DIRECTORY
  launch
  urdf
  worlds
  config
  DESTINATION share/${PROJECT_NAME}
)
```

---

## 4. 대차 모델 구성

### 4.1 기본 링크

현재 대차 모델은 다음 링크와 조인트로 구성합니다.

| 이름 | 종류 | 역할 |
| :--- | :--- | :--- |
| `base_link` | link | 대차 본체 |
| `left_wheel` | link | 왼쪽 바퀴 |
| `right_wheel` | link | 오른쪽 바퀴 |
| `laser_link` | link | LiDAR 센서 위치 |
| `left_wheel_joint` | continuous joint | 왼쪽 바퀴 회전축 |
| `right_wheel_joint` | continuous joint | 오른쪽 바퀴 회전축 |
| `laser_joint` | fixed joint | 본체와 LiDAR 고정 |

### 4.2 Diff Drive 플러그인

`gazebo_ros_diff_drive` 플러그인은 `/cmd_vel`을 받아 좌우 바퀴 조인트를 움직이고 `/odom`을 발행합니다.

```xml
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>/</namespace>
    </ros>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.44</wheel_separation>
    <wheel_diameter>0.16</wheel_diameter>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
    <publish_wheel_tf>true</publish_wheel_tf>
  </plugin>
</gazebo>
```

---

## 5. LiDAR 추가

### 5.1 LiDAR 링크

LiDAR는 `base_link` 위쪽 전방에 고정합니다.

```xml
<link name="laser_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.04"/>
    </geometry>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>

  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.04"/>
    </geometry>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </collision>

  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="laser_joint" type="fixed">
  <parent link="base_link"/>
  <child link="laser_link"/>
  <origin xyz="0.22 0 0.28" rpy="0 0 0"/>
</joint>
```

### 5.2 Gazebo Ray Sensor

Gazebo Classic에서는 `ray` sensor와 `libgazebo_ros_ray_sensor.so` 플러그인을 사용해 LaserScan을 발행합니다.

```xml
<gazebo reference="laser_link">
  <sensor name="laser" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.12</min>
        <max>8.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>laser_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

---

## 6. 실행 커맨드

### 6.1 수정 반영 및 시뮬레이션 실행

URDF, launch 파일 수정 후 Gazebo를 다시 실행할 때 사용합니다.

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch agv_sim sim.launch.py
```

### 6.2 수동 주행

키보드로 `/cmd_vel`을 발행합니다.

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

명령으로 직접 속도를 발행합니다.

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.0}}" -r 10
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.5}}" -r 10
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" --once
```

### 6.3 토픽 확인

토픽 목록, 타입, 주요 센서 데이터를 확인합니다.

```bash
ros2 topic list
ros2 topic list -t
ros2 topic echo /scan
ros2 topic echo /odom
ros2 topic echo /cmd_vel
ros2 topic hz /scan
ros2 topic info /scan
```

### 6.4 TF 확인

`odom -> base_link -> laser_link` 연결을 확인합니다.

```bash
ros2 run tf2_ros tf2_echo base_link laser_link
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_tools view_frames
```

### 6.5 Robot Description 확인

현재 실행 중인 URDF가 의도한 모델인지 확인합니다.

```bash
ros2 param get /robot_state_publisher robot_description
ros2 param get /robot_state_publisher robot_description | grep laser_link
ros2 param get /robot_state_publisher robot_description | grep gazebo_ros_ray_sensor
```

### 6.6 RViz 확인

RViz에서 `RobotModel`, `TF`, `LaserScan`, `Odometry`, `Map` display를 확인합니다.

```bash
rviz2
```

### 6.7 SLAM 실행 및 지도 저장

LiDAR와 TF가 정상일 때 SLAM Toolbox를 실행하고 지도를 저장합니다.

```bash
ros2 launch slam_toolbox online_async_launch.py
ros2 run nav2_map_server map_saver_cli -f my_map
```

### 6.8 ROS2 그래프 확인

실행 중인 node와 ROS2 환경 상태를 확인합니다.

```bash
ros2 node list
ros2 node info /robot_state_publisher
ros2 doctor
```

---

## 7. 목표 토픽 구조

자율주행 대차 시뮬레이션에서 최소로 필요한 토픽과 frame은 다음과 같습니다.

### 7.1 주요 토픽

| 토픽 | 타입 | 역할 |
| :--- | :--- | :--- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 로봇 속도 명령 |
| `/odom` | `nav_msgs/msg/Odometry` | 로봇 위치 추정 |
| `/scan` | `sensor_msgs/msg/LaserScan` | LiDAR 거리 데이터 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 동적 좌표 변환 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 정적 좌표 변환 |
| `/joint_states` | `sensor_msgs/msg/JointState` | 조인트 상태 |

### 7.2 주요 TF 구조

```text
map -> odom -> base_link -> laser_link
```

초기 Gazebo 주행 단계에서는 최소한 아래 구조가 나와야 합니다.

```text
odom -> base_link -> laser_link
```

SLAM과 Navigation2 단계에서는 `map -> odom`이 추가됩니다.

---

## 8. 다음 단계

1. RViz에서 LaserScan 시각화
2. SLAM Toolbox로 지도 작성
3. 지도 저장 후 Navigation2 bringup 연결
4. Nav2 parameter tuning
5. 장애물 회피 및 경로 추종 알고리즘 실험


