import pandas as pd
# import yaml
import math
from ruamel.yaml import YAML
import ast

yaml = YAML()
yaml.preserve_quotes = True

def generate_yaml_file(file, global_planner, controller):
    # copying base yaml file
    base_file = 'base/' + global_planner + '_param_nav2.yaml'
    with open(base_file, 'r') as src_file:
        content_base = yaml.load(src_file)

    # copying controller yaml file
    base_file = 'base/' + str(controller) + '_controller.yaml'
    with open(base_file, 'r') as src_file:
        content_controller = yaml.load(src_file)
    
    # creating file
    with open(file, 'w') as dest_file:
        yaml.dump(content_base, dest_file)
        yaml.dump(content_controller, dest_file)
    
        
def edit_yaml_file (file, footprint_type, footprint, inflation_cost_scale, inflation_radius_global, inflation_radius_local):
    with open(file, 'r') as src_file:
        data = yaml.load(src_file)
    
    ros_params_global = data['global_costmap']['global_costmap']['ros__parameters']
    ros_params_local = data['local_costmap']['local_costmap']['ros__parameters']
    inflation_layer_global = data['global_costmap']['global_costmap']['ros__parameters']['inflation_layer']
    inflation_layer_local = data['local_costmap']['local_costmap']['ros__parameters']['inflation_layer']

    if footprint_type == 0:
        ros_params_global['robot_radius'] =  float(footprint)
        ros_params_local['robot_radius'] =  float(footprint)
        radius = footprint
        
    else:
        ros_params_global['footprint'] = str(footprint)
        ros_params_local['footprint'] =  footprint
        # radius = calc_inflation_radius(footprint)
        radius = incircle_radius_tangential_polygon(ast.literal_eval(footprint))

    inflation_layer_global['cost_scaling_factor'] =  inflation_cost_scale
 
    # inflation_layer_global['inflation_radius'] =  float(radius) * (1 + inflation_radius_global)  
    inflation_layer_global['inflation_radius'] =  inflation_radius_global

    inflation_layer_local['cost_scaling_factor'] =  inflation_cost_scale
    # inflation_layer_local['inflation_radius'] =  float(radius) * (1 + inflation_radius_local)
    inflation_layer_local['inflation_radius'] =   inflation_radius_local


    with open(file, 'w') as dest_file:
         yaml.dump(data, dest_file)
         print("File", file, "created")

def calc_inflation_radius(footprint):
    footprint=yaml.load(footprint)
    return round(math.sqrt((float(footprint[1][0]) -float(footprint[2][0]))**2 + (float(footprint[1][1])-float(footprint[2][1]))**2), 2)


def polygon_area(points):
    """
    Calculates the signed area using the Shoelace formula.
    points: list of (x, y) pairs, in order (clockwise or CCW).
    """
    area = 0.0
    n = len(points)
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
    A = abs(polygon_area(points))
    P = polygon_perimeter(points)
    if P == 0:
        return 0.0
    return 2.0 * A / P


def main():
    csv_file = 'parameters_updated.csv'
    df = pd.read_csv(csv_file)
    for row in df.itertuples(): 
        # getting data
        global_planner = row.Global_Planner
        footprint_type = row.Footprint_type
        footprint = row.Footprint
        controller = row.Controller
        inflation_cost_scale = row.Inflation_cost_scale 

        inflation_radius_global = row.Inflation_radius # 0.1, 0.5, 0.8
        inflation_radius_local = row.Inflation_radius # 0.1, 0.5, 0.8
        #inflation_radius_global = row.Inflation_radius_global # 0.1, 0.5, 0.8
        #inflation_radius_local = row.Inflation_radius_local # 0.1, 0.5, 0.8

        # generating yaml file           
        yaml_file_name = 'generated/it_' + str(row.Index + 1) +'_param_nav2.yaml'
        generate_yaml_file(yaml_file_name, global_planner, controller)        
        edit_yaml_file(yaml_file_name, footprint_type, footprint, inflation_cost_scale, inflation_radius_global, inflation_radius_local)


if __name__ == "__main__":
    main()
