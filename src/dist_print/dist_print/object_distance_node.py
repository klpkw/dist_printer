import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from tracked_object_msgs.msg import ObjectArray, Object
from geometry_msgs.msg import Pose, Vector3, Twist, Accel, Point, Quaternion
from nav_msgs.msg import Path
from math import sqrt, atan2, sin, cos
from tf_transformations import euler_from_quaternion
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
            distance_to_center = self.calculate_distance(car_position, object_position)
            distance_to_bounding_box = self.calculate_bounding_box_distance(
                car_position, self.car_pose.orientation, obj.pose, obj.dimensions)

            self.get_logger().info(
                f'Object ID: {obj.id}, '
                f'Type: {self.get_classification_name(obj.classification)}, '
                f'Distance to center: {distance_to_center:.2f} meters, '
                f'Distance to nearest point: {distance_to_bounding_box:.2f} meters'
            )

        # Print entity counts
        self.print_entity_counts(entity_counter)

    def calculate_distance(self, pos1, pos2):
        return sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2)

    def calculate_bounding_box_distance(self, car_position, car_orientation, object_pose, dimensions):
        # Assuming the orientation is in quaternion form, convert it to yaw angle
        yaw = self.get_yaw_from_quaternion(car_orientation)

        # Get the size of the robot (assuming width is x and height is y)
        robot_size = Vector3(x=1.0, y=0.5)  # Example size of the robot

        # Calculate the distance to the edge of the robot from its center
        distance_to_robot_edge_x = robot_size.x / 2
        distance_to_robot_edge_y = robot_size.y / 2

        # Assuming object dimensions are the width and height of the obstacle
        obstacle_width, obstacle_height = dimensions.x, dimensions.y

        # Calculate the distance to the edge of the obstacle from its center
        distance_to_obstacle_edge_x = obstacle_width / 2
        distance_to_obstacle_edge_y = obstacle_height / 2

        # Calculate the distance between the centers of the robot and the obstacle
        distance_between_centers = self.calculate_distance(car_position, object_pose.position)

        # Rotate the coordinates of the obstacle to match the robot's orientation
        rotated_obstacle_center = self.rotate_point(object_pose.position, car_position, -yaw)

        # Calculate the distance between the edges of the robot and the obstacle
        dx = abs(rotated_obstacle_center.x - car_position.x) - distance_to_robot_edge_x - distance_to_obstacle_edge_x
        dy = abs(rotated_obstacle_center.y - car_position.y) - distance_to_robot_edge_y - distance_to_obstacle_edge_y
        dx = max(dx, 0)
        dy = max(dy, 0)
        return sqrt(dx * dx + dy * dy)

    def get_yaw_from_quaternion(self, orientation):
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        euler = euler_from_quaternion(quaternion)
        return euler[2]  # Yaw angle

    def rotate_point(self, point, origin, angle):
        rotated_x = cos(angle) * (point.x - origin.x) - sin(angle) * (point.y - origin.y) + origin.x
        rotated_y = sin(angle) * (point.x - origin.x) + cos(angle) * (point.y - origin.y) + origin.y
        return Point(x=rotated_x, y=rotated_y, z=point.z)

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

    def print_entity_counts(self, entity_counter):
        for classification, count in entity_counter.items():
            self.get_logger().info(f'{self.get_classification_name(classification)}: {count}')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
