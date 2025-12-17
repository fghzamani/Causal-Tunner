import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import json
import os
import ast
import math
#############################################This filew is for post processing the data we collected in the tials to calculate the local and global path scores and local and global min distance to obstacle, and create a dataset to combile the input config and the output Nav2 result ################################################################################

csv_file_path = 'parameters_updated.csv'
json_folder_path = '/home/forough/Desktop/causal_navigation/src/causal_discovery/generated_paths/fourth_experiment_koboki/'
output_file_path = 'results_dataset_koboki.csv'


df = pd.read_csv(csv_file_path)
goal_pose = [ 2.64297 , 2.692231] 

global_path_len_weight = 1.0
global_min_dist_to_obstacle_weight = 0.8

local_min_distance_weight = 0.4
local_deviation_weight = 0.2
local_elapsed_time_weight = 0.1
local_reaching_weight = 0.9


combined_data = []

columns = ['Global_Planner', 'Controller', 'Cost_Scaling_Factor', 'Inflation_Radius', 'Global_Path_Score', 'Local_Path_Score','Min_Global_Dist_To_Obst', 'Footprint_Type', 'Footprint_Inscribed_Radius', 'Collision', 'Task_result', 'Min_Local_Dist_To_Obstacl', 'Local_Path_Max_Dev', 'Elapsed_Time', 'Reaching_Error', 'Is_Reached', 'Relaxed_Task_Result']

NavFn=0
SmacHybrid=1
SmacLattice=2
Thetastar=3
DWB=1
MPPI=2
codes = { 'NavFn':0, 'SmacHybrid':1, 'SmacLattice':2, 'Thetastar':3, 'DWB':1, 'MPPI':2 }


def max_deviation(global_path, local_path):
    # Convert lists to NumPy arrays
    global_path = np.array(global_path)
    local_path = np.array(local_path)

    # Build KDTree for fast nearest-neighbor search
    tree = KDTree(global_path)

    # Query nearest neighbor in global path for each point in local path
    distances, _ = tree.query(local_path)

    # Return the maximum deviation
    return np.max(distances)



def polygon_area(points):
    """
    Calculates the signed area using the Shoelace formula.
    points: list of (x, y) pairs, in order (clockwise or CCW).
    """
    area = 0.0
    n = len(points)
    # list_obj = ast.literal_eval(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area

def polygon_perimeter(points):
    """
    Computes the perimeter by summing distances between consecutive vertices.
    """
    perimeter = 0.0
    n = len(points)
    # list_obj = ast.literal_eval(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        perimeter += math.hypot(x2 - x1, y2 - y1)
    return perimeter

def incircle_radius_tangential_polygon(points):
    """
    Returns the inradius for a tangential polygon using 2*Area / Perimeter.
    If the polygon is not tangential, this result may not be a valid 'largest inscribed circle.'
    """
    list_obj = ast.literal_eval(points)
    A = abs(polygon_area(list_obj))
    P = polygon_perimeter(list_obj)
    if P == 0:
        return 0.0
    return 2.0 * A / P
    

def angle_error(target_angle, current_angle):
    """
    Compute the shortest angular difference between two angles in radians.
    
    Args:
        target_angle (float): The desired angle in radians.
        current_angle (float): The current angle in radians.
        
    Returns:
        float: The angular error in radians, in the range [-π, π].
    """
    error = target_angle - current_angle
    # Normalize the angle to [-π, π]
    error = (error + math.pi) % (2 * math.pi) - math.pi
    return error

################################################################
successful_iter = []

goal_pose = (-3.04, 7.44, 3.14) # x,y,yaw


for index, row in df.iterrows():
    json_file_name = f"it_{index + 1}_path_generated.json"  # Assuming the row number is the index + 1 (since index is 0-based)
    json_file_path = os.path.join(json_folder_path, json_file_name)
    
    if os.path.exists(json_file_path):
        # Open and load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        if data.get('is_collided')== False and data.get('task_result')=='succeeded':
            successful_iter.append(index+1)
        # TODO: What did happen to those iterations that the robot got stuck and did not collide but couldn't reach the goal?
        collision = 1 if data.get('is_collided') else 0
        Task_result = 0 if data.get('task_result') == 'failed' else 1  #data.get('task_result') == 'succeeded'
        global_path_wp = []
        for p in data.get('path_global_planner'):
            global_path_wp.append([p.get('pose')[0] , p.get('pose')[1]])
            
               
        Global_path_length = data.get('global_path_length')
        global_path_score = -1* global_path_len_weight * Global_path_length + global_min_dist_to_obstacle_weight * data.get("min_dist_to_obstacle")
        
        local_path_wp = []
        min_local_dist_to_obstacle = 1000000.0
        is_reached = False
        relaxed_task_result = 0
        for p in data.get('path_with_controller'):
            lin_vel = p.get('linear_velocity')
            ang_vel = p.get('angular_velocity')
            min_lidar_val = p.get('min_scan_value')
            footprint_cost = p.get('footprint_cost')
            local_path_wp.append([p.get('pose')[0] , p.get('pose')[1]])
            
            orientation_error = angle_error(math.pi, p.get('pose')[2])
            reaching_error = np.linalg.norm(np.array([goal_pose[0],goal_pose[1]]) - np.array([p.get('pose')[0] , p.get('pose')[1]]))
            
            if abs(orientation_error) < 0.9 and reaching_error < 0.5:
                is_reached = True
                print("in the relaxed task result condition 8888888888")
            
            if min_local_dist_to_obstacle > min_lidar_val:
                min_local_dist_to_obstacle = min_lidar_val
        
        
    
        if is_reached:
            relaxed_task_result = 1
            
        
        reaching_error = np.linalg.norm(np.array(local_path_wp[-1]) - np.array(global_path_wp[-1]))
        
        local_path_max_dev = max_deviation(global_path_wp, local_path_wp)
        
        elapsed_time = data.get("duration_of_moving")
        # TODO: Double check with the causal_inference_test file and the way there socers are calculated 
        local_path_score = local_min_distance_weight * min_local_dist_to_obstacle + \
                            local_deviation_weight * (-local_path_max_dev)/10.0 + \
                            local_elapsed_time_weight * (-elapsed_time)/25.0 + \
                            local_reaching_weight * (-reaching_error)
                            
                            
                            
            
        global_planner = codes[row['Global_Planner']] 
        Controller = codes[row['Controller']]
        Footprint_type = row['Footprint_type']
        Inflation_radius = row['Inflation_radius']
        Footprint = row['Footprint']
        Inflation_cost_scale = row['Inflation_cost_scale']
        Global_path_length = data.get('global_path_length')
        Local_path_length = data.get('local_path_length')
        
        combined_row = {
            'Global_Planner': global_planner,
            'Controller': Controller, 
            'Cost_Scaling_Factor': Inflation_cost_scale,
            'Inflation_Radius': Inflation_radius,
            'Global_Path_Score': global_path_score ,
            'Local_Path_Score': local_path_score,
            'Min_Global_Dist_To_Obst': data.get("min_dist_to_obstacle"),
            'Footprint_Type': Footprint_type ,
            'Footprint_Inscribed_Radius': incircle_radius_tangential_polygon(Footprint) if Footprint_type == 1 else Footprint , 
            'Collision': collision,
            'Task_result': Task_result ,
            'Min_Local_Dist_To_Obstacl': min_local_dist_to_obstacle,
            'Local_Path_Max_Dev': local_path_max_dev,
            'Elapsed_Time': elapsed_time,
            'Reaching_Error': reaching_error,
            'Relaxed_Task_Result': relaxed_task_result,
            'Is_Reached': is_reached
            
        }
        combined_data.append(combined_row)
    else:
        print( "in the else with iteration: ", json_file_path )
print(successful_iter)   
# Convert combined data to DataFrame
output_df = pd.DataFrame(combined_data, columns=columns)

weights = {
    'min_dist_to_obstacl': 0.3,
    'local_path_max_dev': 0.2,
    'elapsed_time': 0.3,
    'reaching_error': 0.2
}

# List of columns to normalize
cols_to_normalize = [ "Global_Path_Score", "Local_Path_Score"] # "Local_Path_Max_Dev", "Elapsed_Time", "Reaching_Error",

# Normalize each column using min–max normalization and store the normalized values in new columns
for col in cols_to_normalize:
    min_val = output_df[col].min()
    max_val = output_df[col].max()
    # Check to avoid division by zero in case all values are equal
    if max_val - min_val == 0:
        output_df[col + '_norm'] = 0.0
    else:
        output_df[col + '_norm'] = (output_df[col] - min_val) / (max_val - min_val)

# Compute the weighted sum of the normalized columns row-wise

# output_df['Local_Path_Score'] = (
#     weights['min_dist_to_obstacl'] * output_df['Min_Local_Dist_To_Obstacl_norm'] +
#     weights['local_path_max_dev'] * output_df['Local_Path_Max_Dev_norm'] +
#     weights['elapsed_time'] * output_df['Elapsed_Time_norm'] +
#     weights['reaching_error'] * output_df['Reaching_Error_norm']
# )
# Optionally, if you no longer need the intermediate normalized columns, you can drop them:
output_df['Global_Path_Score'] = output_df['Global_Path_Score_norm']
output_df['Local_Path_Score'] = output_df['Local_Path_Score_norm']
output_df.drop(columns=[col + '_norm' for col in cols_to_normalize], inplace=True)

# Print the resulting DataFrame
print(output_df)

print("Data has been combined and saved.")

final_df = output_df.copy()
# final_df.drop(columns=['Cost Scaling Factor','Inflation Radius','Footprint Inscribed Radius','Local Path Length','Task_result','min_dist_to_obstacl','local_path_max_dev','elapsed_time','reaching_error'], inplace=True)

final_df.to_csv(output_file_path, index=False)
