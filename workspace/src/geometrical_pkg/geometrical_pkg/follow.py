#!/usr/bin/env python3
"""
ROS 2 Node to drive a mobile robot (e.g., TurtleBot3)
to follow predefined paths such as a square, a polygon, or a circle,
using odometry feedback.
"""
import rclpy                                                                # type: ignore
from rclpy.node import Node                                                 # type: ignore
import threading
import math
import time

from geometry_msgs.msg import Twist                                         # type: ignore
from nav_msgs.msg import Odometry                                           # type: ignore
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy    # type: ignore


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def quaternion_to_yaw(qx, qy, qz, qw):
    """
    Calculates the yaw (rotation around Z) from a quaternion.

    Args:
        qx (float): X component of the quaternion.
        qy (float): Y component of the quaternion.
        qz (float): Z component of the quaternion.
        qw (float): W component of the quaternion.

    Returns:
        float: Yaw angle in radians.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle):
    """
    Normalizes an angle to the range [-pi, pi).

    Args:
        angle (float): The angle in radians.

    Returns:
        float: The normalized angle in radians.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


# -------------------------------------------------------------------
# Main ROS Node
# -------------------------------------------------------------------
class Turtlebot3SquarePath(Node):
    """
    A ROS 2 node that controls a TurtleBot3 to execute basic geometric paths 
    using odometry feedback.

    Handles the odometry subscription and velocity command publication.
    """
    def __init__(self):
        super().__init__('turtlebot3_square_path_node')

        # QoS for odometry and velocity
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers/Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)

        # Parameters
        self.rate_hz = 20.0

        # State variables for odometry
        self.x = None
        self.y = None
        self.yaw = None

        # Control flags
        self.is_running = True

        # Measurements for the grapevine rows
        self.straight_length = 3.825
        self.side_length = 1.050

        self.get_logger().info('Node initialized.')

    # -------------------------------------------------------------------
    # Callbacks and Support Methods
    # -------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        """
        Extracts the odometry pose (x, y, yaw) from the Odometry message.

        Args:
            msg (Odometry): ROS 2 Odometry message.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def publish_twist(self, linear=0.0, angular=0.0):
        """
        Publishes linear and angular velocity commands to the /cmd_vel topic.

        Args:
            linear (float): Linear velocity in X (m/s).
            angular (float): Angular velocity in Z (rad/s).
        """
        t = Twist()
        t.linear.x = float(linear)
        t.angular.z = float(angular)
        self.cmd_pub.publish(t)

    def publish_stop(self):
        """Publishes a zero velocity command (stops the robot)."""
        self.publish_twist(0.0, 0.0)

    def wait_for_odom(self, timeout=5.0):
        """
        Waits until the first odometry coordinates (x, y, yaw) are received.

        Args:
            timeout (float): Maximum wait time in seconds.

        Returns:
            bool: True if odometry was received, False otherwise.
        """
        start = time.time()
        while rclpy.ok() and (self.x is None or self.yaw is None):
            if time.time() - start > timeout:
                return False
            time.sleep(0.05)
        return True

    # -------------------------------------------------------------------
    # Movement Primitives
    # -------------------------------------------------------------------
    def move_distance(self, distance, linear_vel):
        """
        Moves forward or backward for a given distance using odometry for feedback.

        Movement stops when the target distance has been covered.

        Args:
            distance (float): Distance to travel (m). Positive for forward, negative for backward.
            linear_vel (float): Maximum linear velocity (m/s).
        """
        if not self.wait_for_odom():
            self.get_logger().warn('No odom — aborting move_distance().')
            return

        start_x, start_y = self.x, self.y
        target = abs(distance)
        direction = 1.0 if distance >= 0 else -1.0
        rate_dt = 1.0 / self.rate_hz

        self.get_logger().info(f'Moving {("forward" if direction > 0 else "backward")} {target:.2f} m')

        while rclpy.ok() and self.is_running:
            dx = self.x - start_x
            dy = self.y - start_y
            traveled = math.hypot(dx, dy)

            if traveled >= target:
                break

            remaining = max(target - traveled, 0.0)
            # Proportional control to decelerate at the end
            speed = min(linear_vel, remaining * 1.0 + 0.05)

            self.publish_twist(linear=direction * speed)
            time.sleep(rate_dt)

        self.publish_stop()
        time.sleep(0.05)

    def rotate_angle(self, angle_rad, angular_vel):
        """
        Rotates by a specified angle (rad) using yaw feedback from odometry.

        Rotation stops when the target angle is reached.

        Args:
            angle_rad (float): Rotation angle in radians. Positive for CCW.
            angular_vel (float): Maximum angular velocity (rad/s).
        """
        if not self.wait_for_odom():
            self.get_logger().warn('No odom — aborting rotate_angle().')
            return

        start_yaw = self.yaw
        target_yaw = normalize_angle(start_yaw + angle_rad)
        direction = 1.0 if angle_rad >= 0 else -1.0
        rate_dt = 1.0 / self.rate_hz

        self.get_logger().info(f'Rotating {math.degrees(angle_rad):.1f} degrees')

        while rclpy.ok() and self.is_running:
            if self.yaw is None:
                time.sleep(rate_dt)
                continue

            error = normalize_angle(target_yaw - self.yaw)
            if abs(error) < math.radians(1.5): # 1.5 degree tolerance
                break

            k = 1.2 # Proportional Gain (P)
            angular_speed = min(angular_vel, max(0.05, abs(k * error)))

            self.publish_twist(angular=direction * angular_speed)
            time.sleep(rate_dt)

        self.publish_stop()
        time.sleep(0.05)

    # -------------------------------------------------------------------
    # High-level Trajectories
    # -------------------------------------------------------------------
    def run_square_path(self, side_length, duration, per_rot_duration=0.3):
        """
        Executes a square path with 4 sides.

        Linear and angular velocities are calculated based on the total duration.

        Args:
            side_length (float): Length of one side of the square (m).
            duration (float): Total desired duration of the path (s).
            per_rot_duration (float): Fraction of total duration dedicated to rotation
                                      (e.g., 0.3 means 30% of total time).
        """
        if not self.wait_for_odom(timeout=10.0):
            self.get_logger().error('No odom received — cannot start square.')
            return

        self.get_logger().info('Starting square path.')

        # TODO
        
        self.get_logger().info('Completed square path.')


    def run_grapevines_row(self, straight_length, side_length, linear_vel=0.4, angular_vel=0.4):
        """
        Navigates a serpentine path through grapevine rows using odometry.

        The robot follows a 'U-turn' pattern to cover parallel rows. It moves down a row, 
        transitions to the next via a 90-degree side maneuver, and repeats the process. 
        By the end of the two iterations, the robot will have covered four rows and 
        returned to its initial heading.

        Args:
            straight_length (float): The length of each grapevine row in meters.
            side_length (float): The distance between two parallel rows in meters.
            linear_vel (float): Forward velocity in m/s. Defaults to 0.4.
            angular_vel (float): Rotational velocity in rad/s. Defaults to 0.4.

        Returns:
            None
        """
        if not self.wait_for_odom(timeout=10.0):
            self.get_logger().error('No odom received — cannot start row navigation.')
            return

        self.get_logger().info('Starting to follow grapevine rows...')

        # TODO
        
        self.get_logger().info('Completed grapevine row navigation.')

# -------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------
def main(args=None):
    """
    Main function to run the node.
    Initializes ROS 2, creates the node, spins it in a separate thread,
    and executes the desired trajectory.
    """
    rclpy.init(args=args)
    node = Turtlebot3SquarePath()

    # Spin in background so odom updates continuously
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # Example calls:
        # node.run_square_path(side_length=1.0, duration=12.0)
        node.run_grapevines_row(node.straight_length, node.side_length)
    except KeyboardInterrupt:
        pass
    finally:
        node.is_running = False
        node.publish_stop()

        node.get_logger().info('Shutting down.')

        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
