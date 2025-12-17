#! /usr/bin/env python3
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav2_msgs.srv import GetCostmap
from nav2_simple_commander.robot_navigator import BasicNavigator
from causal_discovery.footprint_collision_checker import *
from rcl_interfaces.srv import SetParameters, GetParameters
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from nav2_msgs.srv import GetCostmap
from geometry_msgs.msg import Polygon, Point32
from tf_transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
import matplotlib.pyplot as plt
from PIL import Image

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter

import os
import sys
import csv
import json
import time
import math
import copy
import time
import numpy as np
import subprocess
import matplotlib.pyplot as plt


def real_time_plot(costmap, robot_positions, resolution=0.05, update_interval=0.01):
    """
    Real-time plotting of the costmap and robot's position.

    Args:
        costmap (np.ndarray): 2D array representing the costmap.
        robot_positions (list of tuples): List of (y, x) positions for the robot.
        resolution (float): Resolution of the map (for scaling if needed).
        update_interval (float): Time between updates in seconds.
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initial map plot
    im = ax.imshow(costmap, cmap='gray_r', origin='lower')
    robot_plot, = ax.plot([], [], 'ro', markersize=10, label='Robot')  # Empty plot for robot

    ax.set_title('Real-Time Costmap with Robot Position')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Cost')

    for robot_pos in robot_positions:
        y_map, x_map = robot_pos

        # Update robot position
        robot_plot.set_data(x_map, y_map)

        plt.draw()
        plt.pause(update_interval)  # Pause to create the real-time effect

    plt.ioff()  # Turn off interactive mode
    plt.show()
    
    

def min_distance_to_obstacle(costmap, robot_pos , resolution = 0.05):
    """
    Calculate the minimum Euclidean distance from the robot to the nearest obstacle.

    Args:
        costmap (np.ndarray): 2D array where 255 indicates an obstacle.
        robot_pos (tuple): (x, y) position of the robot.

    Returns:
        float: Minimum distance to the nearest obstacle.
    """
    # Step 1: Find obstacle positions
    obstacle_positions = np.argwhere(costmap > 253)  # shape: (num_obstacles, 2)
    # Step 2: Subtract robot position to get relative coordinates
    diffs = obstacle_positions - np.array(robot_pos)  # Broadcasting robot position
    # Step 3: Calculate Euclidean distances
    dists = np.linalg.norm(diffs, axis=1)
    # Step 4: Return the minimum distance
    return np.min(dists)*resolution

def save_dict_to_file(dict_, file_path):
    # Save the dictionary to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(dict_, json_file, indent=4) 


def load_dict_from_file(file_path):
    # Load the dictionary back from the JSON file
    with open(file_path, "r") as json_file:
        loaded_data = json.load(json_file)
    return loaded_data


map_file = "/home/forough/Desktop/causal_navigation/src/pmb2_navigation/pmb2_maps/config/causal_navigation_house.pgm"  # Path to your .pgm file
map_resolution = 0.05  # Map resolution in meters/pixel
map_origin = (
    -6.01,
    -5.01,
)  # Origin of the map in meters (bottom-left corner in map frame)
rotation_angle = 0
map_data = {}
map_data["origin_x"] = map_origin[0]
map_data["origin_y"] = map_origin[1]
map_data["resolution"] = map_resolution
map_data["size_in_cell_x"] = 216
map_data["size_in_cell_y"] = 304


def load_and_rotate_map_image(file_path, rotation_angle):
    # Load the map image using PIL
    img = Image.open(file_path)
    img = img.convert("L")  # Convert to grayscale

    # Rotate the image
    rotated_img = img.rotate(rotation_angle, expand=True)

    return np.array(rotated_img)


def adjust_origin_for_rotation(origin, map_shape, resolution, rotation_angle):
    # Adjust the origin based on the rotation angle
    if rotation_angle == 90:
        # New origin is the bottom-right corner of the original map
        new_origin_x = origin[0] + map_shape[1] * resolution
        new_origin_y = origin[1]
    elif rotation_angle == 180:
        # New origin is the top-right corner of the original map
        new_origin_x = origin[0] + map_shape[1] * resolution
        new_origin_y = origin[1] + map_shape[0] * resolution
    elif rotation_angle == 270:
        # New origin is the top-left corner of the original map
        new_origin_x = origin[0]
        new_origin_y = origin[1] + map_shape[0] * resolution
    else:
        # No rotation
        new_origin_x, new_origin_y = origin

    return new_origin_x, new_origin_y


def world_to_map_coords(x, y, resolution, origin, map_shape, rotation_angle):
    # Convert world (map) coordinates to image (pixel) coordinates
    mx = int((x - origin[0]) / resolution)
    my = int((-1 * y + origin[1]) / resolution) + map_shape[0]
    # my = int((y - origin[1]) / resolution)
    # Adjust coordinates based on rotation
    if rotation_angle == 90:
        mx, my = my, map_shape[1] - mx
    elif rotation_angle == 180:
        mx, my = map_shape[1] - mx, map_shape[0] - my
    elif rotation_angle == 270:
        mx, my = map_shape[0] - my, mx

    return mx, my


def plot_path_on_rotated_map(
    map_image,
    resolution,
    origin,
    rotation_angle,
    color="red",
    path=None,
    controller_path=None,
    iteration_num=None,
):
    # Plot the rotated map
    plt.figure(figsize=(8, 8))
    plt.imshow(map_image, cmap="gray", origin="upper")

    # Extract path coordinates
    path_x = []
    path_y = []
    path_controller_x = []
    path_controller_y = []

    for pose_stamped in path.poses:
        # Convert world coordinates to map (pixel) coordinates
        map_x, map_y = world_to_map_coords(
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            resolution,
            origin,
            map_image.shape,
            rotation_angle,
        )
        path_x.append(map_x)
        path_y.append(map_y)

    for pose in controller_path:
        map_x, map_y = world_to_map_coords(
            pose["pose"][0],
            pose["pose"][1],
            resolution,
            origin,
            map_image.shape,
            rotation_angle,
        )
        path_controller_x.append(map_x)
        path_controller_y.append(map_y)

    # Overlay the path on the map
    plt.plot(
        path_x,
        path_y,
        marker="*",
        color=color,
        linewidth=1,
        markersize=3,
        label="GPlanner Path",
    )
    plt.plot(
        path_controller_x,
        path_controller_y,
        marker="*",
        color="blue",
        linewidth=1,
        markersize=3,
        label="controller Path",
    )
    plt.legend()
    plt.title("Rotated Map with Planned Path")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    # plt.show()
    save_dir = (
        "/home/forough/Desktop/causal_navigation/src/causal_discovery/params_generator/evaluation_results/plots/"
    )
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)  # Ensure directory exists
    save_path = os.path.join(save_dir, f"plot_{str(iteration_num)}.png")
    plt.savefig(save_path)
    plt.close()
    print("end of the plot on the map")


def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a quaternion (x, y, z, w).

    Parameters:
    - roll: Rotation around the x-axis in radians
    - pitch: Rotation around the y-axis in radians
    - yaw: Rotation around the z-axis in radians

    Returns:
    - A tuple (x, y, z, w) representing the quaternion
    """
    # Compute half angles
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)


class CostmapClient(Node):
    def __init__(self):
        super().__init__("costmap_client")

        # Create a client for the GetCostmap service
        self.get_costmap_client = self.create_client(
            GetCostmap, "/global_costmap/get_costmap"
        )
        self.costmap = None
        self.metadata = None
        self.costmap_response = None
        # Wait for the service to become available
        while not self.get_costmap_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /global_costmap/get_costmap service...")

        # Create the request
        self.get_costmap_request = GetCostmap.Request()

    def get_costmap(self):
        # Call the service
        future = self.get_costmap_client.call_async(self.get_costmap_request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.process_costmap(future.result())
        else:
            self.get_logger().error(
                "Failed to call /global_costmap/get_costmap service."
            )

    def process_costmap(self, costmap_response):
        # Extract metadata and costmap data
        self.metadata = costmap_response.map.metadata
        self.costmap = costmap_response.map.data
        self.costmap_response = costmap_response
        # Print metadata
        # self.get_logger().info(f"Resolution: {metadata.resolution}")
        # self.get_logger().info(f"Size: {metadata.size_x} x {metadata.size_y}")
        # self.get_logger().info(f"Origin: ({metadata.origin.position.x}, {metadata.origin.position.y})")

        # # Print sample costmap values
        # size_x = metadata.size_x
        # size_y = metadata.size_y

        self.get_logger().info("Costmap processing complete.")


class FootprintClient(Node):
    def __init__(self):
        super().__init__("footprint_client")

        # Create a parameter client for the global costmap
        self.param_client = self.create_client(
            GetParameters,
            "/global_costmap/global_costmap/get_parameters",
        )
        # Wait for the service to become available
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for parameter service...")

    def get_footprint(self):
        # Create the request to get the 'footprint' parameter
        request = GetParameters.Request()
        request.names = ["footprint"]

        # Call the service and wait for the response
        future = self.param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.values:
                # Extract the footprint value
                footprint = response.values[0].string_value
                if footprint == "[]":  # Check if the footprint is empty
                    self.get_logger().info(
                        "Footprint is empty. Falling back to robot_radius."
                    )
                    return self.get_robot_radius_as_footprint()
                else:
                    self.get_logger().info(f"Footprint: {footprint}")
                    return footprint
            else:
                self.get_logger().error("Footprint parameter not found!")
                return None
        else:
            self.get_logger().error("Failed to call parameter service!")
            return None

    def get_robot_radius_as_footprint(self):
        # Create the request to get the 'robot_radius' parameter
        request = GetParameters.Request()
        request.names = ["robot_radius"]

        # Call the service and wait for the response
        future = self.param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.values:
                robot_radius = response.values[0].double_value
                self.get_logger().info(f"Robot radius: {robot_radius}")

                # Generate a circular footprint based on the radius
                footprint = self.generate_circular_footprint(robot_radius)
                self.get_logger().info(f"Generated circular footprint: {footprint}")
                return footprint
            else:
                self.get_logger().error("Robot radius parameter not found!")
                return None
        else:
            self.get_logger().error("Failed to call parameter service!")
            return None

    def generate_circular_footprint(self, radius, num_points=16):
        """Generate a circular footprint as a polygon."""
        points = [
            (
                radius * math.cos(2 * math.pi * i / num_points),
                radius * math.sin(2 * math.pi * i / num_points),
            )
            for i in range(num_points)
        ]
        return points


class LaserScanSubscriber(Node):
    def __init__(self):
        super().__init__("laser_scan_subscriber")

        # Subscribe to laser scan data
        self.scan_subscription = self.create_subscription(
            LaserScan, "/scan_raw", self.scan_callback, 10  # Laser scan topic
        )

        self.min_range = None  # Store minimum scan value

    def scan_callback(self, msg):
        """Process laser scan data and get the minimum range when the robot is moving."""
        # Remove invalid readings (range < min_range or range > max_range)
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.min_range = min(valid_ranges)  # Store minimum scan value
        else:
            self.min_range = None  # No valid readings

    def get_min_scan(self):
        """Return the latest minimum scan value."""
        return self.min_range


robot_feedback_x, robot_feedback_y, robot_feedback_yaw = 0, 0, 0


def main():
    global robot_feedback_x, robot_feedback_y, robot_feedback_yaw
    rclpy.init()

    navigator = BasicNavigator()

    navigator.declare_parameter("csv_file", "path.csv")
    navigator.declare_parameter("iteration_num", 9)
    navigator.declare_parameter("initial_pose",[0.0,0.0,0.0])
    navigator.declare_parameter("goal_pose", [0.0,0.0,0.0])
    navigator._initial_pose = list(navigator.get_parameter("initial_pose").get_parameter_value().double_array_value)
    # Set our demo's initial pose
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = "map"
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = navigator._initial_pose[0]
    initial_pose.pose.position.y = navigator._initial_pose[1]
    x, y, z, w = euler_to_quaternion(0, 0, navigator._initial_pose[-1]) # -1.57
    initial_pose.pose.orientation.z = z  # sin(π/4)
    initial_pose.pose.orientation.w = w  # cos(π/4) rotate robot to help the controller does not get stuck at the begining
    initial_pose.pose.orientation.x = x  # cos(π/4) rotate robot to help the controller does not get stuck at the begining
    initial_pose.pose.orientation.y = y  # cos(π/4) rotate robot to help the controller does not get stuck at the begining

    # navigator.setInitialPose(initial_pose)
    
    
    def odom_callback(msg):
        """Check if the robot is moving by looking at its velocity."""
        global linear_velocity, angular_velocity
        linear_velocity = msg.twist.twist.linear
        angular_velocity = msg.twist.twist.angular

        
    navigator.odom_subscription = navigator.create_subscription(
        Odometry, "/mobile_base_controller/odom", odom_callback, 10  # Odometry topic
    )

    def amcl_callback(msg):
        global robot_feedback_x, robot_feedback_y, robot_feedback_yaw
        robot_feedback_x = msg.pose.pose.position.x
        robot_feedback_y = msg.pose.pose.position.y

        orientation = msg.pose.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, robot_feedback_yaw = euler_from_quaternion(quaternion)
        

    navigator.amcl_subscription = navigator.create_subscription(
        PoseWithCovarianceStamped, "/amcl_pose", amcl_callback, 5  # QoS depth
    )
    navigator.lifecycleStartup()
    navigator.waitUntilNav2Active()
    navigator.setInitialPose(initial_pose)
    time.sleep(5)

    node = CostmapClient()
    cost_map = node.get_costmap()
    global_costmap = node.costmap_response
    node.destroy_node()

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "map"
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    navigator.goal_pose = list(navigator.get_parameter("goal_pose").get_parameter_value().double_array_value)
    print("navigator.goal_pose ###############", list(navigator.goal_pose))
    goal_pose.pose.position.x = navigator.goal_pose[0]
    goal_pose.pose.position.y = navigator.goal_pose[1]
    goal_pose.pose.orientation.z = 1.0  #TODO: get the value from list and 
    goal_pose.pose.orientation.w = 0.0

    

    # Get the parameter value (the path to the CSV file)
    navigator.csv_file = (
        navigator.get_parameter("csv_file").get_parameter_value().string_value
    )
    navigator.iteration_num = (
        navigator.get_parameter("iteration_num").get_parameter_value().integer_value
    )

    path = navigator.getPath(initial_pose, goal_pose)

    ########## check for the footprint possible collisions on the path  ###########################
    # footprint_client = FootprintClient()
    # footprint = footprint_client.get_footprint()
    # if type(footprint) == str:
    #     footprint = json.loads(footprint)
    # footprint_client.destroy_node()
    footprint = [[0.3, 0.3], [-0.3, 0.3], [-0.3, -1.3], [0.3, -1.3]]
    footprint_polygon = Polygon()
    for point in list(footprint):
        p = Point32()
        p.x = float(point[0])
        p.y = float(point[1])
        p.z = 0.0  # Usually set z to 0 unless you need 3D
        footprint_polygon.points.append(p)
    footprint_checker = FootprintCollisionChecker()
    footprint_checker.set_mapData(map_data)

    footprint_checker.setCostmap(global_costmap)
    x_i = initial_pose.pose.position.x
    y_i = initial_pose.pose.position.y
    path_global_planner = []
    previous_pose = initial_pose.pose.position
    global_path_length = 0
    robot_poses = []
    min_dist_to_obstacle = 10000.0
    for pose_stamped in path.poses:
        pose = pose_stamped.pose
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z

        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        delta_x = x - x_i
        delta_y = y - y_i
        x_i = x
        y_i = y
        orientation = np.arctan2(delta_y, delta_x)
        footprint_cost = footprint_checker.footprintCostAtPose(
            x, y, orientation, footprint_polygon
        )
        

        p1 = previous_pose
        p2 = pose.position
        dist = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        global_path_length += dist
        previous_pose = copy.copy(pose.position)
        

        resolution = global_costmap.map.metadata.resolution
        w = global_costmap.map.metadata.size_x
        h = global_costmap.map.metadata.size_y
        (x_map, y_map) = footprint_checker.worldToMapValidated(x,y)
        # robot_pos =  np.array([x_map, y_map])
        robot_pos =  np.array([y_map, x_map])
        map = np.array(global_costmap.map.data)
        robot_poses.append(robot_pos)
        
        map = map.reshape((h, w))
        min_distance = min_distance_to_obstacle(robot_pos=robot_pos , costmap= map , resolution = resolution)
        if min_distance < min_dist_to_obstacle:
            min_dist_to_obstacle = min_distance
     
        path_global_planner.append(
            {"pose": [x, y, orientation], "footprint_cost": footprint_cost , "min_dist_to_obstacle": min_distance}
        )
          
    # real_time_plot(map , robot_positions=robot_poses , resolution=0.05)

    smoothed_path = navigator.smoothPath(path)

    # Follow path
    navigator.followPath(smoothed_path)
    t1 = time.time()

    i = 0
    path_with_controller = []
    is_collided = False
    prev_pose = [initial_pose.pose.position.x, initial_pose.pose.position.y]
    local_path_length = 0
    node = LaserScanSubscriber()

    # rclpy.spin(node)
    while not navigator.isTaskComplete():
        i += 1
        feedback = navigator.getFeedback()
        # if feedback and i % 5 == 0:
        # print(
        #     "Estimated distance remaining to goal position: "
        #     + "{0:.3f}".format(feedback.distance_to_goal)
        #     + "\nCurrent speed of the robot: "
        #     + "{0:.3f}".format(feedback.speed)
        # )

        footprint_cost = footprint_checker.footprintCostAtPose(
            robot_feedback_x,
            robot_feedback_y,
            robot_feedback_yaw,
            footprint_polygon,
        )
        local_dist = math.sqrt(
            (robot_feedback_x - prev_pose[0]) ** 2
            + (robot_feedback_y - prev_pose[1]) ** 2
        )
        local_path_length += local_dist
        rclpy.spin_once(node)
        min_scan_value = node.get_min_scan()
        path_with_controller.append(
            {
                "pose": [robot_feedback_x, robot_feedback_y, robot_feedback_yaw],
                "footprint_cost": footprint_cost,
                "linear_velocity": [
                    linear_velocity.x,
                    linear_velocity.y,
                    linear_velocity.z,
                ],
                "angular_velocity": [
                    angular_velocity.x,
                    angular_velocity.y,
                    angular_velocity.z,
                ],
                "min_scan_value": min_scan_value,
            }
        )
        prev_pose = [robot_feedback_x, robot_feedback_y]
        print("footprint_cost-------------", footprint_cost)
        if footprint_cost > 253:
            r_inscribed = np.sqrt(0.93**2 + (0.2-0.11)**2)
            if min_scan_value < (r_inscribed + 0.1) :
                navigator.cancelTask()
                is_collided = True
                break
    t2 = time.time()
    node.destroy_node()
    print("is_colide at the end of the while ::::::::", is_collided)
    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED and not is_collided:
        print("Goal succeeded!")
        task_result = "succeeded"
    elif result == TaskResult.CANCELED:
        print("Goal was canceled!")
        task_result = "canceled"

    elif result == TaskResult.FAILED or (
        result == TaskResult.SUCCEEDED and is_collided
    ):
        print("Goal failed!")
        task_result = "failed"

    else:
        print("Goal has an invalid return status!")
        task_result = "invalid_status"
    duration_of_moving = t2 - t1
    result_dict = {
        "path_global_planner": path_global_planner,
        "path_with_controller": path_with_controller,
        "task_result": task_result,
        "iteration_num": navigator.iteration_num,
        "is_collided": is_collided,
        "global_path_length": global_path_length,
        "local_path_length": local_path_length,
        "duration_of_moving": duration_of_moving,
        "min_dist_to_obstacle":min_dist_to_obstacle, # min distance to obstacle for the global path
    }
    save_dict_to_file(result_dict, navigator.csv_file)

    map_image = load_and_rotate_map_image(map_file, rotation_angle)

    # Adjust the origin based on the rotation
    new_map_origin = adjust_origin_for_rotation(
        map_origin, map_image.shape, map_resolution, rotation_angle
    )
    # Plot the path on the rotated map using the adjusted origin

    plot_path_on_rotated_map(
        map_image,
        map_resolution,
        new_map_origin,
        rotation_angle,
        path=path,
        controller_path=path_with_controller,
        iteration_num=navigator.iteration_num,
    )

    time.sleep(0.5)
    navigator.lifecycleShutdown()
    # terminate_ros_processes()
    return


def terminate_ros_processes():
    pkill_commands = [
        "pkill -9 -f 'ros2'",
        "pkill -9 -f 'gazebo'",
        "pkill -9 -f 'gz'",
        "pkill -9 -f 'nav'",
    ]

    for cmd in pkill_commands:
        print(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True, check=True)  # Execute command


if __name__ == "__main__":
    main()
