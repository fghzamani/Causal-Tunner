#!/usr/bin/env python3


import os
from time import sleep
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    pkg_pal_nav2_dir = get_package_share_directory('pmb2_2dnav')
    pkg_robot_simulation = get_package_share_directory('robot_simulation')

    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    autostart = LaunchConfiguration('autostart', default='True')

    nav2_launch_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_pal_nav2_dir, 'launch', 'pmb2_nav_bringup.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'map': os.path.join(pkg_robot_simulation, 'maps', 'my_map.yaml')
        }.items()
    )

    rviz_launch_cmd = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            '-d' + os.path.join(
                get_package_share_directory('nav2_bringup'),
                'rviz',
                'nav2_default_view.rviz'
            )]
    )

    # amcl_init_pose_publisher
    set_init_amcl_pose_cmd = Node(
        package="robot_simulation",
        executable="init_amcl_pose_publisher", 
        name="init_amcl_pose_publisher",
        parameters=[{
            "x": 0,
            "y": 0,
        }]
    )

    demo_cmd = Node(
        package='robot_simulation',
        executable='send_pose_nav2',
        emulate_tty=True,
        output='screen',
    )
    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(nav2_launch_cmd)
    ld.add_action(set_init_amcl_pose_cmd)
    ld.add_action(rviz_launch_cmd)
    ld.add_action(demo_cmd)
    
    
    return ld