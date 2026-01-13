# Causal-Tunner

This repository provides tools for **tuning robot navigation parameters** using causal inference in simulation. It is organized into the following main components:

- **`robot_simulation/`**: ROS2 package for running Gazebo simulations with configurable navigation parameters
- **`causal_discovery/`**: Python package for causal discovery, parameter tuning, and data collection
- **`plasys_house_world/`**: Gazebo world models for simulation environments

---

## Directory Structure

```
Causal-Tunner/
│
├── robot_simulation/        # ROS2 package for running Gazebo simulations
│   ├── launch/
│   ├── maps/
│   ├── models/
│   ├── params/
│   ├── worlds/
│   └── ...                  # ROS2 nodes and package files
│
├── causal_discovery/        # Causal discovery and parameter tuning
│   ├── causal_discovery/    # Python package
│   │   ├── __init__.py
│   │   ├── evaluation_paths.py
│   │   ├── footprint_collision_checker.py
│   │   ├── get_footprint.py
│   │   └── path_plotter_navigator.py
│   ├── generated_paths/     # Generated path data (JSON files)
│   ├── generated_plots/     # Generated visualization plots
│   ├── params_generator/    # Parameter generation scripts
│   ├── data_collector.sh    # Main script to run simulations and collect data
│   ├── setup.py
│   ├── setup.cfg
│   └── package.xml
│
├── plasys_house_world/      # Gazebo world files and models
│
├── dependencies.rosinstall  # ROS2 dependencies file
├── LICENSE
└── README.md
```

---

## Installation

To set up the workspace and install dependencies:

```bash
mkdir -p ~/causal_tunner_ws/src
cd ~/causal_tunner_ws/src
git clone https://github.com/fghzamani/Causal-Tunner.git
cd ~/causal_tunner_ws
vcs import src < src/Causal-Tunner/dependencies.rosinstall
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

---

## Environment Setup

Add the following lines to your `~/.bashrc` file:

```bash
# Set your workspace path
export ROS_CAUSAL_WS="$HOME/causal_tunner_ws"

# Gazebo paths for simulation models
export GAZEBO_MODEL_PATH=$ROS_CAUSAL_WS/src/Causal-Tunner/robot_simulation/models:$GAZEBO_MODEL_PATH
export GAZEBO_RESOURCE_PATH=$ROS_CAUSAL_WS/src/Causal-Tunner/robot_simulation:$GAZEBO_RESOURCE_PATH
```

Then reload your bashrc:

```bash
source ~/.bashrc
```

---

## Usage

### 1. Running Navigation Trials and Collecting Data

Run the data collector script to run simulations with different parameter iterations and collect data:

```bash
bash $ROS_CAUSAL_WS/src/Causal-Tunner/causal_discovery/data_collector.sh
```

This script will:
- Run multiple simulation iterations in Gazebo
- Generate paths and collect navigation data
- Save results to JSON files in the `generated_paths/` folder

---

### 2. Causal Discovery

To analyze the collected data and discover causal relationships:

```bash
cd $ROS_CAUSAL_WS/src/Causal-Tunner/causal_discovery
python3 causal_discovery.py
```

This script will read your CSV data, build a causal graph, and output results/figures to the `figures/` folder.

---

### 3. Parameter Inference and Tuning

For inferring optimal navigation parameters using causal models, run the relevant Python scripts in `causal_discovery/`:

```bash
cd $ROS_CAUSAL_WS/src/Causal-Tunner/causal_discovery/scripts
python3 causal_inference.py
```

These scripts will use the discovered causal relationships to suggest parameter configurations that optimize navigation outcomes.

---

## Dependencies

- ROS2 (Humble/Iron)
- Gazebo
- Python 3.x
- pandas, numpy, networkx, pgmpy, scikit-learn, matplotlib

---

## Customization

- Adjust the number of trials, parameter ranges, and experiment settings in the config files and scripts as needed
- Automate batch experiments or parameter sweeps using the provided bash scripts

---

## Demonstration with Real Robot (Mirte Master)

https://github.com/user-attachments/assets/65ff1920-044f-4813-a3f8-31d9cab7c5ea

---

## License

Apache-2.0 License (see `LICENSE` file).

---

## Contact

For questions or issues, please open a GitHub issue.
