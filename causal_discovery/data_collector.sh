#!/bin/bash

# Check if ROS_CAUSAL_WS environment variable is defined
if [ -z "$ROS_CAUSAL_WS" ]; then
    echo "ERROR: ROS_CAUSAL_WS environment variable is not defined."
    echo "Please set it before running this script:"
    echo "  export ROS_CAUSAL_WS=\"\$HOME/Desktop/ros2_ws\""
    exit 1
fi

# Verify the workspace path exists
if [ ! -d "$ROS_CAUSAL_WS" ]; then
    echo "ERROR: Workspace path does not exist: $ROS_CAUSAL_WS"
    exit 1
fi

echo "Using workspace: $ROS_CAUSAL_WS"

# Detect shell and source appropriate setup file
if [ -n "$ZSH_VERSION" ]; then
    source "$ROS_CAUSAL_WS/install/setup.zsh"
else
    source "$ROS_CAUSAL_WS/install/setup.bash"
fi
 # Run the first launch file
for i in {1..1792}
    do
        echo "Iteration $i"
        echo "Running simulation" &
        ros2 launch robot_simulation world_launch.py > a.txt & 
        
        # Set the current date and time as part of the bag file namep 
        # bag_filename="rosbag_$(date +"%Y"-"%m"-"%d"_"%H"_"%M")"


        # Give some time for the first launch file to initialize
        echo "Nav parameter file: $ROS_CAUSAL_WS/src/causal_discovery/params_generator/generated_configs/it_${i}_param_nav2.yaml" &
        sleep 4

        # Run the second launch file with rosbag recording
        echo "Running navigation" 
        ros2 launch pal_navigation_cfg_bringup nav_bringup.launch.py map:=$ROS_CAUSAL_WS/src/Causal-Tunner/robot_simulation/maps/map.yaml nav2_param:="$ROS_CAUSAL_WS/src/causal_discovery/params_generator/generated/it_${i}_param_nav2.yaml" > b.txt &
        sleep 1
        PID1=$!
        echo "PID :"
        echo  $PID1
	sleep 6
	
        # Run the rosnode
        echo "Running causal discovery"
        ros2 run causal_discovery path_plotter_navigator --ros-args -p csv_file:="$ROS_CAUSAL_WS/src/Causal-Tunner/causal_discovery/generated_paths/it_${i}_path_generated.json" -p iteration_num:="${i}"
        # PID2=$!

        sleep 2
        # Start rosbag recording on all topics
        # ros2 bag record -a -o $bag_filename
        
        # Wait for user to interrupt the script (Ctrl+C) to stop recording and close the launched processes 
        # trap "killall -SIGINT rosbag record && killall -SIGINT ros2 && pkill -9 gz" INT
        
        # The process end after path generation how
        
        #echo "Iteration $i finished execution" &  
        #pkill -9 -f "ros" & pkill -9 -f "gazebo" & pkill -9 -f "gz" &  pkill -9 -f "nav" &
        
        kill -INT $PID1 
        source $ROS_CAUSAL_WS/src/Causal-Tunner/causal_discovery/kill_ros_process.sh
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
        # source ~/Downloads/killall.sh
        # kill -INT $PID2
        # pkill -9 -f 'nav'
        # sleep 3
        # if ps -p "$PID1" > /dev/null; then
        # echo "Process $PID1 did not terminate with SIGINT, sending SIGTERM"
        # kill -TERM "$PID1"
        # sleep 2
        # fi
        echo "End of iteration $i" 
       
done
