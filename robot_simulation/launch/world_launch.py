# #!/usr/bin/env python3

# import os
# from os import environ, pathsep

# from ament_index_python.packages import get_package_share_directory, get_package_prefix
# from launch import LaunchDescription
# from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import Node
# from launch_pal.include_utils import include_launch_py_description
# from launch.conditions import IfCondition



# def get_model_paths(packages_names):
#     model_paths = ''
#     for package_name in packages_names:
#         if model_paths != '':
#             model_paths += pathsep

#         package_path = get_package_prefix(package_name)
#         model_path = os.path.join(package_path, 'share')

#         model_paths += model_path

#     return model_paths

# def get_real_model_paths(packages_names):
#     model_paths = ''
#     for package_name in packages_names:
#         if model_paths != '':
#             model_paths += pathsep
#         package_path = get_package_prefix(package_name)
#         model_path = os.path.join(package_path, 'share', 'robot_simulation', 'models')
#         for (path, directories, filenames) in os.walk(model_path):
#             for directory in directories:
#                 file_path = os.path.join(path, directory)
#                 model_paths += file_path
#                 model_paths += pathsep
#     return model_paths

# def get_resource_paths(packages_names):
#     resource_paths = ''
#     for package_name in packages_names:
#         if resource_paths != '':
#             resource_paths += pathsep

#         package_path = get_package_prefix(package_name)
#         resource_paths += package_path

#     return resource_paths


# def generate_launch_description():
#     pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
#     # pkg_configuration = get_package_share_directory('configuration')
#     pkg_robot_simulation = get_package_share_directory('robot_simulation')
    
#     use_sim_time = LaunchConfiguration('use_sim_time', default='true')
#     x_pose = LaunchConfiguration('x_pose', default='-3.04')
#     y_pose = LaunchConfiguration('y_pose', default='7.44')
#     yaw_pose = LaunchConfiguration('Yaw_pose', default='-1.57')

#     world = os.path.join(
#         pkg_robot_simulation,
#         'worlds',
#         'AH_store_3.world'
#     )

#     gzserver_cmd = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
#         ),
#         launch_arguments={'world': world}.items()
#     )

#     gzclient_cmd = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
#         )
#     )

#     # robot_state_publisher_cmd = IncludeLaunchDescription(
#     #     PythonLaunchDescriptionSource(
#     #         os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
#     #     ),
#     #     launch_arguments={'use_sim_time': use_sim_time}.items()
#     # )


#     # tiago_gazebo_spawner_cmd = Node(
#     #         package='tiago_gazebo',
#     #         executable='tiago_spawn.launch.py',
#     #         name='tiago_gazebo',
#     #         output='screen'
#     #     )

#     navigation_arg = DeclareLaunchArgument(
#         'navigation', default_value='false',
#         description='Specify if launching Navigation2'
#     )

#     # moveit_arg = DeclareLaunchArgument(
#     #     'moveit', default_value='false',
#     #     description='Specify if launching MoveIt2'
#     # )
#     artificial_map = Node(
#     package='tf2_ros',
#     executable='static_transform_publisher',
#     output='screen',
#     name = 'link_broadcast',
#     arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'odom'], #check names
#     )


#     # world_name_arg = DeclareLaunchArgument(
#     #     'world_name', default_value='pal_office',
#     #     description="Specify world name, we'll convert to full path"
#     # )


#     tiago_spawn = include_launch_py_description(
#         'tiago_gazebo', ['launch', 'robot_spawn.launch.py'],
#         launch_arguments={'use_sim_time': 'True', 'x_pose': x_pose, 'yaw_pose': yaw_pose, "y_pose": y_pose, 'robot_name': 'tiago','laser_model':'sick-571' }.items()) #'arm':['left-arm'], 'arm_type':'no-arm' laser_model:''

#     tiago_bringup = include_launch_py_description(
#         'tiago_bringup', ['launch', 'tiago_bringup.launch.py'],
#         launch_arguments={'use_sim_time': 'True'}.items(), 
#         )

#     navigation = include_launch_py_description(
#         'tiago_2dnav', ['launch', 'tiago_sim_nav_bringup.launch.py'],
#         launch_arguments={'use_sim_time': 'True'}.items(),
#         condition=IfCondition(LaunchConfiguration('navigation')))

#     # move_group = include_launch_py_description(
#     #     'tiago_moveit_config', ['launch', 'move_group.launch.py'],
#     #     launch_arguments={'use_sim_time': 'True'}.items(),
#     #     condition=IfCondition(LaunchConfiguration('moveit')))

#     # tuck_arm = Node(package='tiago_gazebo',
#     #                 executable='tuck_arm.py',
#     #                 emulate_tty=True,
#     #                 output='both',
#     #                 parameters=[{'use_sim_time': True}])


#     packages = ['tiago_description', 'pmb2_description',
#                 'hey5_description', 'pal_gripper_description']
#     model_path = get_model_paths(packages)
#     resource_path = get_resource_paths(packages)
#     if 'GAZEBO_MODEL_PATH' in environ:
#         model_path += pathsep + environ['GAZEBO_MODEL_PATH']
#         model_path += pathsep + get_real_model_paths(['robot_simulation'])

#     if 'GAZEBO_RESOURCE_PATH' in environ:
#         resource_path += pathsep + environ['GAZEBO_RESOURCE_PATH']

    

#     ld = LaunchDescription()
#     ld.add_action(SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path))
#     ld.add_action(gzserver_cmd)
#     ld.add_action(gzclient_cmd)
#     ld.add_action(tiago_spawn)
#     ld.add_action(tiago_bringup)

#     # ld.add_action(navigation_arg)
#     # ld.add_action(navigation)
#     ld.add_action(artificial_map)
    


#     # ld.add_action(moveit_arg)
#     # ld.add_action(move_group)
 
#     # ld.add_action(tuck_arm)
#     # ld.add_action(start_gazebo_ros_spawner_cmd)
#     # ld.add_action(tiago_gazebo_spawner_cmd)
#     return ld


#!/usr/bin/env python3

import os
from os import environ, pathsep

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_pal.include_utils import include_launch_py_description
from launch.conditions import IfCondition



def get_model_paths(packages_names):
    model_paths = ''
    for package_name in packages_names:
        if model_paths != '':
            model_paths += pathsep

        package_path = get_package_prefix(package_name)
        model_path = os.path.join(package_path, 'share')

        model_paths += model_path

    return model_paths

def get_real_model_paths(packages_names):
    model_paths = ''
    for package_name in packages_names:
        if model_paths != '':
            model_paths += pathsep
        package_path = get_package_prefix(package_name)
        model_path = os.path.join(package_path, 'share', 'plasys_house_world','plasys_house_world', 'models')
        for (path, directories, filenames) in os.walk(model_path):
            for directory in directories:
                file_path = os.path.join(path, directory)
                model_paths += file_path
                model_paths += pathsep
    return model_paths

def get_resource_paths(packages_names):
    resource_paths = ''
    for package_name in packages_names:
        if resource_paths != '':
            resource_paths += pathsep

        package_path = get_package_prefix(package_name)
        resource_paths += package_path

    return resource_paths


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    # pkg_configuration = get_package_share_directory('configuration')
    pkg_robot_simulation = get_package_share_directory('robot_simulation')
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    declare_x_arg = DeclareLaunchArgument(
        'x_pose', default_value='0.0',
        description='Initial robot X position'
    )
    declare_y_arg = DeclareLaunchArgument(
        'y_pose', default_value='0.0',
        description='Initial robot Y position'
    )
    declare_yaw_arg = DeclareLaunchArgument(
        'yaw_pose', default_value='1.57',
        description='Initial robot yaw (radians)'
    )
    
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    yaw_pose = LaunchConfiguration('yaw_pose')
    
    # world = os.path.join(
    #     pkg_robot_simulation,
    #     'worlds',
    #     # 'AH_store_3.world'
    #     'KRR_Course_Small_house.world'
    # )
    world = os.path.join(get_package_share_directory('plasys_house_world'), 'worlds','plasys_house', 'causal_navigation_house.world') 
    

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # robot_state_publisher_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
    #     ),
    #     launch_arguments={'use_sim_time': use_sim_time}.items()
    # )


    # tiago_gazebo_spawner_cmd = Node(
    #         package='tiago_gazebo',
    #         executable='tiago_spawn.launch.py',
    #         name='tiago_gazebo',
    #         output='screen'
    #     )

    navigation_arg = DeclareLaunchArgument(
        'navigation', default_value='false',
        description='Specify if launching Navigation2'
    )

    # moveit_arg = DeclareLaunchArgument(
    #     'moveit', default_value='false',
    #     description='Specify if launching MoveIt2'
    # )
    artificial_map = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    output='screen',
    name = 'link_broadcast',
    arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'odom'], #check names
    )


    # world_name_arg = DeclareLaunchArgument(
    #     'world_name', default_value='pal_office',
    #     description="Specify world name, we'll convert to full path"
    # )


    tiago_spawn = include_launch_py_description(
        'tiago_gazebo', ['launch', 'robot_spawn.launch.py'],
        launch_arguments={'use_sim_time': 'True', 'x_pose': x_pose, 'y_pose': y_pose, 'yaw_pose': yaw_pose, 'robot_name':'tiago', 'laser_model':'no-laser' }.items()) #'arm':['left-arm'], 'arm_type':'no-arm'

    tiago_bringup = include_launch_py_description(
        'tiago_bringup', ['launch', 'tiago_bringup.launch.py'],
        launch_arguments={'use_sim_time': 'True'}.items(), 
        )

    navigation = include_launch_py_description(
        'tiago_2dnav', ['launch', 'tiago_sim_nav_bringup.launch.py'],
        launch_arguments={'use_sim_time': 'True'}.items(),
        condition=IfCondition(LaunchConfiguration('navigation')))

    # move_group = include_launch_py_description(
    #     'tiago_moveit_config', ['launch', 'move_group.launch.py'],
    #     launch_arguments={'use_sim_time': 'True'}.items(),
    #     condition=IfCondition(LaunchConfiguration('moveit')))

    # tuck_arm = Node(package='tiago_gazebo',
    #                 executable='tuck_arm.py',
    #                 emulate_tty=True,
    #                 output='both',
    #                 parameters=[{'use_sim_time': True}])


    packages = ['tiago_description', 'pmb2_description',
                'hey5_description', 'pal_gripper_description']
    model_path = get_model_paths(packages)
    resource_path = get_resource_paths(packages)
    if 'GAZEBO_MODEL_PATH' in environ:
        model_path += pathsep + environ['GAZEBO_MODEL_PATH']
        model_path += pathsep + get_real_model_paths(['robot_simulation'])

    if 'GAZEBO_RESOURCE_PATH' in environ:
        resource_path += pathsep + environ['GAZEBO_RESOURCE_PATH']

    

    ld = LaunchDescription()
    ld.add_action(SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path))
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(tiago_spawn)
    ld.add_action(tiago_bringup)

    # ld.add_action(navigation_arg)
    # ld.add_action(navigation)
    ld.add_action(artificial_map)
    ld.add_action(declare_x_arg)
    ld.add_action(declare_y_arg)
    ld.add_action(declare_yaw_arg)

    # ld.add_action(moveit_arg)
    # ld.add_action(move_group)
 
    # ld.add_action(tuck_arm)
    # ld.add_action(start_gazebo_ros_spawner_cmd)
    # ld.add_action(tiago_gazebo_spawner_cmd)
    return ld




