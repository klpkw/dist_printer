import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from tracked_object_msgs.msg import ObjectArray, Object
from geometry_msgs.msg import Pose, Vector3, Twist, Accel
from nav_msgs.msg import Path
from math import sqrt
from collections import defaultdict


class ObjectDistanceNode(Node):

    def __init__(self):
        super().__init__('object_distance_node')
        self.subscription_objects = self.create_subscription(
            ObjectArray,
            '/custom_tracked_objs/lidar',
            self.objects_callback,
            10)
        self.subscription_odometry = self.create_subscription(
            Odometry,
            '/odometry/global',
            self.odometry_callback,
            10)
        self.car_pose = None

    def odometry_callback(self, msg):
        self.car_pose = msg.pose.pose

    def objects_callback(self, msg):
        if self.car_pose is None:
            self.get_logger().info('Waiting for car pose...')
            return

        car_position = self.car_pose.position
        entity_counter = defaultdict(int)

        for obj in msg.objects:
            entity_counter[obj.classification] += 1

            object_position = obj.pose.position
            distance_to_center = self.calculate_distance(
                car_position, object_position)
            distance_to_bounding_box = self.calculate_bounding_box_distance(
                car_position, obj.pose, obj.dimensions)

            self.get_logger().info(
                f'Object ID: {obj.id}, '
                f'Type: {self.get_classification_name(obj.classification)}, '
                f'Distance to center: {distance_to_center:.2f} meters, '
                f'Distance to nearest point: {distance_to_bounding_box:.2f} meters'
            )

        # Print entity counts
        self.print_entity_counts(entity_counter)

    def print_entity_counts(self, entity_counter):
        for classification, count in entity_counter.items():
            self.get_logger().info(
                f'{self.get_classification_name(classification)}: {count}')

    def calculate_distance(self, pos1, pos2):
        return sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2)

    def calculate_bounding_box_distance(self, car_position, object_pose, dimensions):
        dx = abs(car_position.x - object_pose.position.x) - dimensions.x / 2
        dy = abs(car_position.y - object_pose.position.y) - dimensions.y / 2
        dz = abs(car_position.z - object_pose.position.z) - dimensions.z / 2
        dx = max(dx, 0)
        dy = max(dy, 0)
        dz = max(dz, 0)
        return sqrt(dx * dx + dy * dy + dz * dz)

    def get_classification_name(self, classification):
        classification_map = {
            0: 'PEDESTRIAN',
            1: 'CAR',
            2: 'TRUCK',
            3: 'BUS',
            4: 'TRAILER',
            5: 'MOTORCYCLE',
            6: 'BICYCLE',
            99: 'UNKNOWN'
        }
        return classification_map.get(classification, 'UNKNOWN')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
