from setuptools import find_packages, setup

package_name = 'causal_discovery'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='forough',
    maintainer_email='fgh.zamani72@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_plotter_navigator = causal_discovery.path_plotter_navigator: main',
            'evaluation_paths = causal_discovery.evaluation_paths: main', 
            'footprint_collison_checker = causal_discovery.footprint_collision_checker: main',
            'get_footprint = causal_discovery.get_footprint: main'
        ],
    },
)
