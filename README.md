# Causal-Tunner

This repository provides tools for **tuning robot navigation parameter ** using causal inference and randomized controlled trials (RCTs) in simulation. It is organized into two main components:

- **`running_rct/`**: A ROS2 package for running navigation trials in Gazebo and collecting experiment data.
- **`causal_inference/`**: A folder containing Python scripts, bash scripts, and figures for causal discovery, inference, and parameter tuning.

---

## Directory Structure

```
causal_navigation_ws/
│
├── running_rct/           # ROS2 package for running trials and collecting data
│   ├── launch/
│   ├── scripts/
│   ├── config/
│   ├── runnig_trials.sh   # Main script to launch trials and save results
│   └── ...                # ROS2 nodes and package files
│
└── causal_inference/      # Causal discovery and parameter tuning
    ├── causal_discovery.py    # Main script for causal discovery
    ├── inferencing.py         # Scripts for parameter inference/tuning
    ├── figures/               # Plots and result figures
    ├── *.sh                   # Bash scripts for automation
    └── ...                    # Additional Python scripts
```

---

## Usage

### 1. **Running Navigation Trials and Collecting Data**

- Use the `running_rct` ROS2 package to run navigation experiments in Gazebo and collect results in a CSV file.
- **Configure paths** in `runnig_trials.sh`:
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
