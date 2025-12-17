# Causal-Tunner

This repository provides tools for **tuning robot navigation parameter ** using causal inference and randomized controlled trials (RCTs) in simulation. It is organized into two main components:

- **`running_rct/`**: A

---

## Directory Structure

```
Causal-Tunner/
│
├── robot_simulation/      # ROS2 package for running Gazebo simulations
│   ├── launch/
│   ├── maps/
│   ├── models/
│   ├── params/
│   ├── worlds/
│   └── ...                # ROS2 nodes and package files
│
├── causal_discovery/      # Causal discovery and parameter tuning
│   ├── causal_discovery/  # Python package
│   │   ├── __init__.py
│   │   ├── evaluation_paths.py
│   │   ├── footprint_collision_checker.py
│   │   ├── get_footprint.py
│   │   └── path_plotter_navigator.py
│   ├── generated_paths/   # Generated    path data (JSON files)
│   ├── generated_plots/   # Generated visualization plots
│   ├── params_generator/  # Parameter generation scripts
│   ├── data_collector.sh  # Main script to run simulations and collect data
│   ├── setup.py
│   ├── setup.cfg
│   └── package.xml
│
├── dependencies.rosinstall  # ROS2 dependencies file
├── LICENSE
└── README.md
```

---

## Installation

To set up the workspace and install dependencies:

```bash
cd ~/my_workspace
vcs import src < src/Causal-Tunner/dependencies.rosinstall
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

---

## Usage

### 1. **Running Navigation Trials and Collecting Data**

Before running the data collection script, set the `ROS_CAUSAL_WS` environment variable to point to your workspace:

```bash
export ROS_CAUSAL_WS="$HOME/Desktop/ros2_ws"
```

Then run the data collector script to run simulations with different parameter iterations and collect data:

```bash
bash causal_discovery/data_collector.sh
```

This script will:
- Run multiple simulation iterations in Gazebo
- Generate paths and collect navigation data
- Save results to JSON files in the `generated_paths/` folder

Alternatively, you can use the `running_rct` ROS2 package to run navigation experiments and collect results in a CSV file:
- **Configure paths** in `running_rct/runnig_trials.sh`:
  - Set the path to your `nav2_param` file.
  - Set the path to your output CSV file.

**Example:**
```bash
cd running_rct
bash runnig_trials.sh
```
This will launch the trials, run the robot in Gazebo, and save experiment results to the specified CSV.

---

### 2. **Causal Discovery**

- To analyze the collected data and discover causal relationships, run:

```bash
cd causal_inference
python3 causal_discovery.py
```

This script will read your CSV data, build a causal graph, and output results/figures to the `figures/` folder.

---

### 3. **Parameter Inference and Tuning**

- For inferring optimal navigation parameters using causal models, run the relevant Python scripts in `causal_inference/`:

```bash
python3 inferencing.py
```
or
```bash
python3 tune_parameters.py
```

- These scripts will use the discovered causal relationships to suggest parameter configurations that optimize navigation outcomes.

---

## Notes

- **Dependencies:**  
  - ROS2
  - Python 3.x, pandas, numpy, networkx, pgmpy, scikit-learn, matplotlib (for `causal_inference`)

---

## Customization

- Adjust the number of trials, parameter ranges, and experiment settings in the config files and scripts as needed.
- You can automate batch experiments or parameter sweeps using the provided bash scripts.

---
 ## Demonstration with real robot
 


https://github.com/user-attachments/assets/65ff1920-044f-4813-a3f8-31d9cab7c5ea


## License

MIT License (see `LICENSE` file).

---

## Contact

For questions or issues, please open a GitHub issue.
