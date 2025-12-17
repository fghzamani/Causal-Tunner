import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

class CostmapSubscriber(Node):
    def __init__(self):
        super().__init__('costmap_subscriber')
        
        # Subscribing to global and local costmap topics
        self.global_costmap_subscriber = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.global_costmap_callback,
            10
        )
        self.local_costmap_subscriber = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.local_costmap_callback,
            10
        )

    def global_costmap_callback(self, msg):
        self.print_costmap_subregion(msg, "Global Costmap", start_x=10, start_y=10, width=5, height=5)

    def local_costmap_callback(self, msg):
        self.print_costmap_subregion(msg, "Local Costmap", start_x=10, start_y=10, width=5, height=5)

    def print_costmap_subregion(self, msg, map_type, start_x, start_y, width, height):
        map_width = msg.info.width
        map_height = msg.info.height

        # Validate the subregion boundaries
        if start_x + width > map_width or start_y + height > map_height:
            self.get_logger().error(f"{map_type}: Subregion is out of bounds!")
            return

        self.get_logger().info(f"{map_type}: Printing subregion starting at ({start_x}, {start_y}) with width {width} and height {height}")

        for y in range(start_y, start_y + height):
            row_values = []
            for x in range(start_x, start_x + width):
                cell_index = y * map_width + x
                if 0 <= cell_index < len(msg.data):
                    cell_value = msg.data[cell_index]
                    row_values.append(cell_value)
                else:
                    row_values.append(None)  # Handle out-of-bounds cells

            # Print the row of values
            self.get_logger().info(f"{map_type}: Row {y} values: {row_values}")

def main(args=None):
    rclpy.init(args=args)
    node = CostmapSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
