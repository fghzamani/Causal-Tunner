#!/bin/bash

source /home/forough/Desktop/causal_navigation/install/setup.bash

# === Load all selected iterations from CSV ===
# iterations=($(tail -n +2 Nav2_random_configurations.csv | cut -d',' -f1))
# echo "Total iterations loaded: ${#iterations[@]}"
# 4 12 20 28 36 452 460 468 476 900 908 916 924 1348 1356 1364 1372 1380
iteration_list=(20)

# resume_index=112 # 27 

# === Slice the array from resume point ===
# iterations=("${iterations[@]:$resume_index}")
# echo "Resuming from iteration index $resume_index (${iterations[0]})"

for i in "${iterations[@]}"; do
# for i in "${iteration_list[@]}";do
    # echo "Iteration $i"
    # echo "Running simulation" &
    # ros2 launch robot_simulation world_launch.py > a.txt & 

    # echo "Nav parameter file: /home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/generated/it_${i}_param_nav2.yaml" &
    # sleep 4

    # echo "Running navigation" 
    # ros2 launch pal_navigation_cfg_bringup nav_bringup.launch.py \
    #     map:=/home/forough/Desktop/causal_navigation/src/pmb2_navigation/pmb2_maps/config/map.yaml \
    #     nav2_param:="/home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/generated/it_${i}_param_nav2.yaml" > b.txt &

    # sleep 1
    # PID1=$!
    # sleep 6

    # echo "Running causal discovery"
    # ros2 run causal_discovery path_plotter_navigator --ros-args \
    #     -p csv_file:="/home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/second trial/generated_paths/it_${i}_path_generated.json" \
    #     -p iteration_num:="${i}"


    echo "Iteration $i"
    echo "Running simulation" &
    ros2 launch robot_simulation world_launch.py > a.txt & 
    
    # Set the current date and time as part of the bag file namep 
    # bag_filename="rosbag_$(date +"%Y"-"%m"-"%d"_"%H"_"%M")"


    # Give some time for the first launch file to initialize
    echo "Nav parameter file: /home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/generated/it_${i}_param_nav2.yaml" &
    sleep 4

    # Run the second launch file with rosbag recording
    echo "Running navigation" 
    ros2 launch pal_navigation_cfg_bringup nav_bringup.launch.py map:=/home/forough/Desktop/causal_navigation/src/pmb2_navigation/pmb2_maps/config/map.yaml nav2_param:="/home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/generated/it_${i}_param_nav2.yaml" > b.txt &
    sleep 1
    PID1=$!
    echo "PID :"
    echo  $PID1
	sleep 6
	
    # Run the rosnode
    echo "Running causal discovery"
    ros2 run causal_discovery path_plotter_navigator --ros-args -p csv_file:="/home/forough/Desktop/causal_navigation/src/causal_discovery/generated_paths/third_experiment/it_${i}_path_generated.json" -p iteration_num:="${i}"
    # PID2=$!
    sleep 2

    kill -INT $PID1
    source ~/Downloads/killall.sh
    sleep 6

    pkill -9 -f 'ros2'
    sleep 1
    pkill -9 -f 'gazebo'
    sleep 1
    pkill -9 -f 'gz'
    sleep 1
    pkill -9 -f 'nav'
    sleep 4

    pkill -9 -f 'ros2'
    pkill -9 -f 'gazebo'
    pkill -9 -f 'gz'
    pkill -9 -f 'nav'
    sleep 3

    echo "End of iteration $i" 
done
