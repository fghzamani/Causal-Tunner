#!/usr/bin/env bash

set -euo pipefail
IFS=$' \n\t'
START_TRIAL=${1:-1}
# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Define your list of pose‐pairs: each element is "x1 y1 yaw1 x2 y2 yaw2"
PAIRS=(
  "-1.5  5.0  -1.57   -6.5  3.0   0.0"
  "-5.0  5.0  -1.57   -1.0  4.0  -1.57" 
  "-6.5  1.5   0.0     0.0  0.0   1.57"
  "-4.0  1.0   0.0    -1.0  5.5  -1.57"  
  "-2.2  2.5   1.57    -6.5  3.0   0.0" 
  " 0.5 -1.0   1.57   -7.5 -2.0   0.0"
  "-6.5  3.0   0.0    -7.0 -2.0   0.0"
  "-1.0  4.0  -1.57  -8.5  4.0   0.0"
  "-8.0  4.0   0.0    -2.2  2.5  -1.57"
  "-1.0  5.6  -1.57     0.0   0.0  1.57"
)


SPAWN_LAUNCH="world_launch.py"
NAV_BRINGUP="nav_bringup.launch.py"
MAP_PATH="/home/forough/Desktop/causal_navigation/src/pmb2_navigation/pmb2_maps/config/causal_navigation_house.yaml"
NAV2_PARAM="/home/forough/Desktop/causal_navigation/src/pal_navigation_cfg_public/pal_navigation_cfg_params/params/marathon2_params.yaml"
# NAV2_PARAM="/home/forough/Desktop/causal_navigation/src/pal_navigation_cfg_public/pal_navigation_cfg_params/params/pmb2_nav2.yaml"
KILLALL_SCRIPT=~/Downloads/killall.sh

# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------

for idx in "${!PAIRS[@]}"; do
  trial=$(( idx + 1 ))

  # Skip all trials before START_TRIAL
  if (( trial < START_TRIAL )); then
    continue
  fi

  # unpack the six numbers into variables
  read -r x1 y1 yaw1 x2 y2 yaw2 <<< "${PAIRS[$idx]}"

  echo
  echo "=== Trial $trial: initial=($x1, $y1, $yaw1) → goal=($x2, $y2, $yaw2) ==="

  # 1) Spawn robot in Gazebo
  echo "[Trial $trial] Spawning robot..."
  ros2 launch robot_simulation $SPAWN_LAUNCH \
    x_pose:=$x1 \
    y_pose:=$y1 \
    yaw_pose:=$yaw1 \
    > a.txt 2>&1 &

  sleep 10

  # 2) Bring up Nav2
  echo "[Trial $trial] Starting navigation stack..."
  ros2 launch pal_navigation_cfg_bringup $NAV_BRINGUP \
    map:=$MAP_PATH \
    nav2_param:=$NAV2_PARAM \
    > b.txt 2>&1 &
  PID_NAV=$!

  sleep 6

  # 3) Run causal discovery evaluation
  echo "[Trial $trial] Running causal discovery..."
  ros2 run causal_discovery evaluation_paths \
    --ros-args \
      -p initial_pose:="[$x1, $y1, $yaw1]" \
      -p goal_pose:="[$x2, $y2, $yaw2]" \
      -p csv_file:="/home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/evaluation_results/paths/mb_it_${trial}_path_generated.json" \
      -p iteration_num:="${trial}"

  sleep 2

  # 4) Tear down everything
  echo "[Trial $trial] Cleaning up..."
  kill -INT $PID_NAV 2>/dev/null || true
  source $KILLALL_SCRIPT

  pkill -9 -f 'ros2'   || true
  pkill -9 -f 'gazebo' || true
  pkill -9 -f 'gz'     || true
  pkill -9 -f 'nav'    || true

  sleep 4
done

echo
echo "Trials ${START_TRIAL} through ${#PAIRS[@]} completed."


# #!/usr/bin/env bash

# set -euo pipefail
# IFS=$' \n\t'

# # Number of random trials
# NUM_TRIALS=11

# # Define your list of poses: each element is "x y yaw" (radians)
# POSES=(
#   "-2.0 5.0 -1.57"
#   "-1.0 4.0 -1.57"
#   "-5.0 5.0 -1.57"
#   "-6.5 3.0 0.0"
#   "-7.5 2.0 0.0"
#   "-7.0 6.5 -1.57"
#   "-4.0 1.0 0.0"
#   "0.0 0.0 1.57"
#   "-2.5 2.0 0.0"
# )


# ##############################################################################

# # Helper: pick a random index in [0, len-1)
# pick_index() {
#   local len=${#POSES[@]}
#   echo $(( RANDOM % len ))
# }

# for (( trial=11; trial<=NUM_TRIALS; trial++ )); do
#   # pick two distinct random indices
#   i1=$(pick_index)
#   i2=$(pick_index)
#   while [[ $i2 -eq $i1 ]]; do
#     i2=$(pick_index)
#   done

#   # read poses
#   read -r x1 y1 yaw1 <<< "${POSES[$i1]}"
#   read -r x2 y2 yaw2 <<< "${POSES[$i2]}"

#   echo "=== Trial $trial: spawn at ($x1, $y1, $yaw1), goal at ($x2, $y2, $yaw2) ==="

#   # 1) Spawn robot in Gazebo at initial pose
#   echo "Running simulation" &
#   ros2 launch robot_simulation world_launch.py x_pose:=$x1 y_pose:=$y1 yaw_pose:=$yaw1 > a.txt &
  
#   sleep 10
#   #ros2 launch pal_navigation_cfg_bringup nav_bringup.launch.py map:=/home/forough/Desktop/causal_navigation/src/pmb2_navigation/pmb2_maps/config/causal_navigation_house.yaml nav2_param:=/home/forough/Desktop/causal_navigation/src/pal_navigation_cfg_public/pal_navigation_cfg_params/params/pmb2_nav2.yaml > b.txt &
#   ros2 launch pal_navigation_cfg_bringup nav_bringup.launch.py map:=/home/forough/Desktop/causal_navigation/src/pmb2_navigation/pmb2_maps/config/causal_navigation_house.yaml nav2_param:=/home/forough/Desktop/testbed0_ws/src/pal_navigation_cfg_public/pal_navigation_cfg_params/params/default_Nav2_param.yaml > b.txt &
    
#   sleep 1
#   PID1=$!
#   echo "PID :"
#   echo  $PID1
#   sleep 6
  
#   echo "Running causal discovery"
#   ros2 run causal_discovery evaluation_paths \
#     --ros-args \
#       -p initial_pose:="[$x1, $y1, $yaw1]" \
#       -p goal_pose:="[$x2, $y2, $yaw2]" \
#       -p csv_file:="/home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/evaluation_results/paths/baseline_it_${trial}_path_generated.json" \
#       -p iteration_num:="${trial}"
#   sleep 2
  
#   # kill -INT $PID1
#   source ~/Downloads/killall.sh
#   sleep 6
  
#   pkill -9 -f 'ros2'
#   sleep 1
#   pkill -9 -f 'gazebo'
#   sleep 1
#   pkill -9 -f 'gz'
#   sleep 1
#   pkill -9 -f 'nav'
#   sleep 4
  
#   pkill -9 -f 'ros2'
#   pkill -9 -f 'gazebo'
#   pkill -9 -f 'gz'
#   pkill -9 -f 'nav'
#   sleep 3
#   echo "All $NUM_TRIALS trials 009090909090 launched."
#   wait
# done

# wait  # wait for all background jobs to finish
# echo "All $NUM_TRIALS trials launched."