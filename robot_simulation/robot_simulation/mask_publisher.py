#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import CostmapFilterInfo

class MyPublisher(Node):

    def __init__(self):
        super().__init__('my_publisher')
        # Publisher for OccupancyGrid
        self.publisher_occupancy_grid = self.create_publisher(OccupancyGrid, 'mask', 1)
        # Publisher for CostmapFilterInfo (assuming you want to publish this as well)
        self.publisher_costmap_filter_info = self.create_publisher(CostmapFilterInfo, 'costmap_filter_info', 1)

        # Set the rate at which to publish the message
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # Create and publish OccupancyGrid message
        mask_value = 10
        occupancy_grid = OccupancyGrid()
        occupancy_grid.info.resolution = 1.0
        occupancy_grid.header.frame_id = 'mask_frame'
        occupancy_grid.info.width = 3
        occupancy_grid.info.height = 3
        occupancy_grid.info.origin.position.x = 3.0
        occupancy_grid.info.origin.position.y = 3.0
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.x = 0.0
        occupancy_grid.info.origin.orientation.y = 0.0
        occupancy_grid.info.origin.orientation.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0
        occupancy_grid.data = [mask_value] * (occupancy_grid.info.width * occupancy_grid.info.height)

        self.publisher_occupancy_grid.publish(occupancy_grid)
        print("hey!!!!!!!!!!!!!!!!")

        # Create and publish CostmapFilterInfo message if needed
        # std::unique_ptr<nav2_msgs::msg::CostmapFilterInfo> msg =
        # std::make_unique<nav2_msgs::msg::CostmapFilterInfo>();
        # msg->type = 0;
        # msg->filter_mask_topic = MASK_TOPIC;
        # msg->base = static_cast<float>(base);
        # msg->multiplier = static_cast<float>(multiplier);
        costmapfilterinfo = CostmapFilterInfo()
        costmapfilterinfo.type = 0
        costmapfilterinfo.filter_mask_topic = "mask"
        costmapfilterinfo.base = 1.0
        costmapfilterinfo.multiplier = 2.0
        self.publisher_costmap_filter_info.publish(costmapfilterinfo)
       

def main(args=None):
    rclpy.init(args=args)
    my_publisher = MyPublisher()
    rclpy.spin(my_publisher)
    my_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()