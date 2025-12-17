#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PolygonStamped, Point32
import matplotlib.pyplot as plt
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import numpy as np
from scipy.spatial import ConvexHull

class FootprintPlotter(Node):
    def __init__(self):
        super().__init__('footprint_plotter')
        
        # Subscriber to the original footprint topic
        self.subscription = self.create_subscription(
            PolygonStamped,
            '/local_costmap/published_footprint',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Publisher for the updated convex footprint
        self.publisher = self.create_publisher(
            PolygonStamped,
            '/local_costmap/published_dynamic_footprint',
            10)

        # TF2 buffer and listener to get gripper_link position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        self.figure, self.ax = plt.subplots()
        self.plot_initialized = False

    def listener_callback(self, msg):
        # Extract the points from the polygon
        points = [ [point.x, point.y] for point in msg.polygon.points ]
        # Get the position of gripper_link
        try:
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id,  # Target frame (same as footprint frame)
                'gripper_link',       # Source frame
                rclpy.time.Time())    # Get the latest available transform

            # Append gripper_link position to the points
            gripper_point = [transform.transform.translation.x, transform.transform.translation.y]
            points.append(gripper_point)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Could not transform gripper_link: {e}')
            return  # Skip this callback if transform is not available

        # Convert points to numpy array
        points_np = np.array(points)

        # Check if we have at least 3 points to compute a convex hull
        if len(points_np) >= 3:
            # Compute the convex hull
            hull = ConvexHull(points_np)
            hull_indices = hull.vertices
            hull_points = points_np[hull_indices]

            # Prepare the new PolygonStamped message
            new_msg = PolygonStamped()
            new_msg.header = msg.header
            new_msg.polygon.points = [Point32(x=pt[0], y=pt[1], z=0.0) for pt in hull_points]
            
            # Close the polygon by adding the first point at the end
            hull_points_closed = np.vstack([hull_points, hull_points[0]])

            # Publish the updated convex footprint
            self.publisher.publish(new_msg)

            # Update plotting data
            x = hull_points_closed[:, 0]
            y = hull_points_closed[:, 1]

        else:
            self.get_logger().warn('Not enough points to compute convex hull.')
            return

        # Clear the previous plot
        self.ax.clear()
        self.ax.plot(x, y, 'b-')  # Plot the convex footprint polygon

        # Set the title and labels
        self.ax.set_title('Convex Robot Footprint with Gripper Link')
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')

        # Optionally, set the axes limits
        margin = 0.5
        self.ax.set_xlim(min(x)-margin, max(x)+margin)
        self.ax.set_ylim(min(y)-margin, max(y)+margin)
        self.ax.set_aspect('equal', 'box')

        # Redraw the plot
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    footprint_plotter = FootprintPlotter()

    try:
        rclpy.spin(footprint_plotter)
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    footprint_plotter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
