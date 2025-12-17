from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_simulation'


def package_files(data_files, directory_list):
    paths_dict = {}
    for directory in directory_list:
        for (path, directories, filenames) in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)
                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]
    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=package_files([
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name),
         glob('launch/*launch.py')),
        (os.path.join('share', package_name, 'maps/'),
         glob('maps/*')),
        (os.path.join('share', package_name, 'worlds/'),
         glob('worlds/*')),
        (os.path.join('share', package_name, 'config/'),
         glob('config/*')),
        (os.path.join('share', package_name, 'params/'),
         glob('params/*'))
    ], ['models/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='forough',
    maintainer_email='forough@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['init_amcl_pose_publisher = robot_simulation.set_initial_pose_amcl: main',
            'remapping = robot_simulation.remapping: main',
            'mask_publisher = robot_simulation.mask_publisher: main',
            'plot_footprint = robot_simulation.plot_footprint: main',
            'costmap_monitor = robot_simulation.costmap_monitor: main'
        
        ],
    },
)


