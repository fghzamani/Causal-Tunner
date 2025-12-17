import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan  # Replace with the actual message type you want to forward




class TopicForwarderNode(Node):
   def __init__(self):
       super().__init__('topic_forwarder_node')
       # Subscribe to the source topic
       self.subscription = self.create_subscription(
           LaserScan,
           '/scan_raw',  # Replace with the source topic
           self.callback,
           10
       )
       # Publish to the destination topic
       self.publisher = self.create_publisher(
           LaserScan,
           '/scan',  # Replace with the destination topic
           10
       )
   def callback(self, msg):
       # Callback function: Publish to the destination topic
       self.publisher.publish(msg)


def main(args=None):
   rclpy.init(args=args)
   node = TopicForwarderNode()
   rclpy.spin(node)
   rclpy.shutdown()
if __name__ == '__main__':
   main()