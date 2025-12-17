import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
import math


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
                    self.get_logger().info("Footprint is empty. Falling back to robot_radius.")
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
            (radius * math.cos(2 * math.pi * i / num_points),
             radius * math.sin(2 * math.pi * i / num_points))
            for i in range(num_points)
        ]
        return points


def main(args=None):
    rclpy.init(args=args)
    node = FootprintClient()
    footprint = node.get_footprint()
    if footprint:
        print(f"Final Footprint: {footprint}")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
