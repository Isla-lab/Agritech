#!/usr/bin/env python3
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import time
import math
import cv2                                                                       # type: ignore
import numpy as np                                                               # type: ignore
import rclpy                                                                     # type: ignore
from rclpy.node import Node                                                      # type: ignore
from sensor_msgs.msg import Image, LaserScan                                     # type: ignore
from nav_msgs.msg import Odometry                                                # type: ignore
from geometry_msgs.msg import Twist                                              # type: ignore
from cv_bridge import CvBridge                                                   # type: ignore
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy         # type: ignore
import random


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__("limo_yolo")

        # ROS PARAMETERS
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("turn_speed_max", 1.5)           # Maximum angular velocity

        self.img_topic = self.get_parameter("image_topic").value
        self.turn_speed_max = self.get_parameter("turn_speed_max").value

        # CAMERA AND IMAGE
        self.bridge = CvBridge()

        # MOVEMENT CONTROL
        self.rate_hz = 20
        self.forward_speed = 0.4        # Linear speed
        self.align_tol = 0.05           # Error tolerance during realignment phase

        # ODOMETRY
        # Current odometry
        self.x = None
        self.y = None
        self.yaw = None

        # Initial odometry
        self.starting_x = None
        self.starting_y = None
        self.starting_yaw = None

        # SEARCH AND SCAN
        self.search_straight_distance = 1 # Straight distance to travel during "FORWARD" phase

        # Starting odometry for the search phase
        self.search_start_x = None
        self.search_start_y = None
        self.search_start_yaw = None

        self.realigning = False

        # OBSTACLE DETECTION
        self.ranges = []

        self.obstacle_detected = False
        self.avoiding = False

        self.obstacle_threshold = 0.4   # Safety distance threshold to trigger avoidance
        self.current_distance = np.inf  # Current distance from any obstacles

        # Starting odometry for obstacle avoidance phase
        self.start_avoiding_yaw = None

        # Stop logic
        self.stopped = False

        # Variables related to checkpoints (intersections)
        self.current_checkpoint = None

        self.checkpoint1_reached = False
        self.checkpoint1_finished = False

        self.checkpoint2_reached = False
        self.checkpoint2_finished = False

        self.checkpoint3_reached = False
        self.checkpoint3_finished = False

        self.checkpoint4_reached = False

        # Checkpoint coordinates
        self.checkpoint1_x = 0.650
        self.checkpoint1_y = -2.000
        self.checkpoint2_x = 0.650
        self.checkpoint2_y = 0.525
        self.checkpoint3_x = -1.825
        self.checkpoint3_y = 0.525

        # Last checkpoint -> final position
        self.checkpoint4_x = -1.825
        self.checkpoint4_y = -2.000

        # Safety measure to ensure the robot does not drift
        # If self.safety_alignment is True, then the robot will periodically realign to the next intersection, avoiding
        # drifting which causes unpredictable and problematic behavior
        # TODO: remove it and see how things change!
        self.safety_alignment = True

        # -----------------------------
        # ROS I/O
        # -----------------------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos)    # Receives odometry
        self.sub_image = self.create_subscription(Image, self.img_topic, self.on_image, 10)     # Receives camera images
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.on_scan, 10)          # Receives LiDAR data
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)                             # Sends velocity commands

        self.control_timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)           # Main control loop

        self.get_logger().info(f"Node loaded...")


    # -----------------------------
    # ODOMETRY CALLBACK
    # -----------------------------
    def odom_callback(self, msg: Odometry):
        """Updates internal pose variables from Odometry messages."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        if self.starting_x is None:
            self.starting_x = self.x
            self.starting_y = self.y
            self.starting_yaw = self.yaw
            self.get_logger().info("Odom initialized")

    # -----------------------------
    # CAMERA CALLBACK
    # -----------------------------
    def on_image(self, msg: Image):
        """Placeholder for camera image processing."""
        return


    # -----------------------------
    # LiDAR CALLBACK
    # -----------------------------
    def on_scan(self, msg: LaserScan):
        """
        LiDAR data callback. Updates the minimum frontal distance 
        by considering a central sector of 40 samples.
        """

        self.ranges = np.array(msg.ranges)

        # TODO
        # Compute self.current_distance as the minimum read range
        # Remember to first clean the data (remove NaNs and Infs)
        # Store also self.current_min_angle, as the angle corresponding to the minimum distance

    # -----------------------------
    # CONTROL METHODS
    # -----------------------------
    def distance_from_point(self, x0: float, y0: float) -> float:
        """
        Calculates the Euclidean distance from the current position (self.x, self.y)
        to a given point (x0, y0).
        """
        return math.hypot(self.x - x0, self.y - y0)
    
    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Calculates the yaw (heading) from a quaternion.
        """
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy*qy + qz*qz)
        return math.atan2(siny, cosy)

    def normalize_angle(self, angle: float) -> float:
        """
        Normalizes an angle to the interval [-pi, pi).
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def angle_error(self, target_angle: float) -> float:
        """
        Calculates the normalized angular error between the target angle and current yaw.
        """
        return self.normalize_angle(target_angle - self.yaw)

    def calculate_target_yaw(self, goal_x: float, goal_y: float) -> float:
        """
        Calculates the yaw angle required to point toward the goal.
        """
        return math.atan2(goal_y - self.y, goal_x - self.x)
    
    def publish_twist(self, linear=0.0, angular=0.0):
        """
        Publishes a linear and angular velocity command to the /cmd_vel topic.

        Args:
            linear (float): Linear velocity in X (m/s).
            angular (float): Angular velocity in Z (rad/s).
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        """Publishes a zero velocity command (stops the robot)."""
        self.publish_twist(0.0, 0.0)

    def print_message(self, message):
        self.get_logger().info(message)
        

    # -----------------------------
    # OBSTACLE DETECTION
    # -----------------------------
    def detect_obstacle(self):
        """
        Checks for the presence of obstacles.

        Prioritizes avoidance: an obstacle is detected if the frontal LiDAR distance
        is below the threshold.
        """
 
        # Detects a generic obstacle if frontal LiDAR distance is too short.
        self.obstacle_detected = self.current_distance < self.obstacle_threshold


    def align_to_next_checkpoint(self):
        """
        Determines the next checkpoint and rotates the robot to face it.
        """
        goal_x = None
        goal_y = None

        # Determine the position of the checkpoint to align to (the one following the reached one)
        if self.current_checkpoint is None:
            goal_x = self.checkpoint1_x
            goal_y = self.checkpoint1_y
        elif self.current_checkpoint == 1:
            goal_x = self.checkpoint2_x
            goal_y = self.checkpoint2_y       
        elif self.current_checkpoint == 2:
            goal_x = self.checkpoint3_x
            goal_y = self.checkpoint3_y    
        elif self.current_checkpoint == 3:
            goal_x = self.checkpoint4_x
            goal_y = self.checkpoint4_y

        # Align with the next checkpoint
        target_yaw = self.calculate_target_yaw(goal_x, goal_y)
        angle_err = self.angle_error(target_yaw)
            
        if abs(angle_err) > self.align_tol:
            # Rotate using proportional control
            angular_vel = max(min(0.8 * angle_err, self.turn_speed_max), -self.turn_speed_max)
            self.publish_twist(0.0, angular_vel)
        else:
            self.realigning = False

        return    

    def check_for_checkpoint_reached(self):
        """
        Checks if the robot has reached the proximity of any checkpoint
        and updates the mission progress.
        """

        # CHECKPOINT_1
        if not self.checkpoint1_reached and not self.checkpoint1_finished:
            d = self.distance_from_point(self.checkpoint1_x, self.checkpoint1_y)
            if d <= 0.1:
                self.checkpoint1_reached = True
                self.checkpoint1_finished = True
                self.current_checkpoint = 1
                return 
            
        # CHECKPOINT_2
        if not self.checkpoint2_reached and not self.checkpoint2_finished:
            d = self.distance_from_point(self.checkpoint2_x, self.checkpoint2_y)
            if d <= 0.1:
                self.checkpoint2_reached = True
                self.checkpoint2_finished = True
                self.current_checkpoint += 1
                return 
            
        # CHECKPOINT_3
        if not self.checkpoint3_reached and not self.checkpoint3_finished:
            d = self.distance_from_point(self.checkpoint3_x, self.checkpoint3_y)
            if d <= 0.1:
                if self.checkpoint4_reached:
                    self.stopped = True     # End of the task
                    self.publish_stop()
                    return
                
                # Else, go toward checkpoint 4
                self.checkpoint3_reached = True
                self.checkpoint3_finished = True
                self.current_checkpoint += 1
                return      
            
        # CHECKPOINT_4
        if not self.checkpoint4_reached:
            d = self.distance_from_point(self.checkpoint4_x, self.checkpoint4_y)
            if d <= 0.1:
                self.checkpoint4_reached = True

                # Head toward checkpoint 3 again
                self.checkpoint3_reached = False
                self.checkpoint3_finished = False
                self.current_checkpoint = 2     # Hardcoded, do not change
                return 


    # -----------------------------
    # ACTION METHODS
    # -----------------------------
    def avoid(self):
        """
        Executes an obstacle avoidance maneuver by checking alternative paths.
        Rotates to look left, then right, then backward until a clear path is found.
        """
        # 1. Initialize Avoidance
        if self.start_avoiding_yaw is None:
            self.start_avoiding_yaw = self.yaw
            self.avoid_step = 0 # 0: Look Left, 1: Look Right, 2: Look Back
            self.avoid_target_yaw = None
            return

        # 2. Define the target angle for the current step
        if self.avoid_target_yaw is None:
            if self.avoid_step == 0:
                target_offset = math.pi / 2   # +90° (Left)
            elif self.avoid_step == 1:
                target_offset = -math.pi / 2  # -90° (Right)
            else:
                target_offset = -math.pi      # -180° (Back)
            
            self.avoid_target_yaw = self.normalize_angle(self.start_avoiding_yaw + target_offset)

        # 3. Rotate toward the target
        angle_error = self.angle_error(self.avoid_target_yaw)

        if abs(angle_error) > 0.1:
            direction = 1.0 if angle_error > 0 else -1.0
            self.publish_twist(0.0, direction * 0.5) # Slightly faster rotation
            return

        # 4. Rotation reached! Now check the path
        self.publish_stop()
        
        if not self.obstacle_detected:
            # PATH CLEAR: Exit avoidance and begin realignment
            self.start_avoiding_yaw = None
            self.avoid_target_yaw = None
            self.avoiding = False
            self.realigning = True

            return
        else:
            # PATH BLOCKED: Try the next direction
            self.avoid_step += 1
            self.avoid_target_yaw = None
            
            # If we've finished looking back and all are blocked, stop the robot
            if self.avoid_step > 2:
                self.stopped = True

    # -----------------------------
    # CONTROL LOOP
    # -----------------------------
    def control_loop(self):
        """
        Main control loop of the robot.

        Manages transitions based on position (checkpoints), 
        safety (obstacle avoidance), and mission logic. Coordinates 
        the execution of navigation and avoidance behaviors.
        """

        if self.x is None or self.yaw is None or self.stopped:
            self.publish_stop()
            return

        # Check if a checkpoint has been reached
        self.check_for_checkpoint_reached()

        # Check if any obstacle is in front
        self.detect_obstacle()

        # If an obstacle is in front and the agent is not avoiding an obstacle already, start avoidance
        if self.obstacle_detected and not self.avoiding:
            self.avoiding = True
            

        if self.avoiding:

            self.avoid()

        else:
            
            if not self.realigning:

                # Save starting odometry for the current segment
                if self.search_start_x is None:
                    self.search_start_y = self.y
                    self.search_start_yaw = self.yaw
                    self.search_start_x = self.x

                # Proceed for predefined straight distance
                d = self.distance_from_point(self.search_start_x, self.search_start_y)

                if d >= self.search_straight_distance:
                    self.publish_stop()

                    # Reset parameters for current segment
                    self.search_start_x = None
                    self.search_start_y = None
                    self.search_start_yaw = None

                    self.realigning = True

                    return

                # Otherwise, continue straight
                self.publish_twist(self.forward_speed, 0.0)
            
            else:
                # Align to next checkpoint (safety constraint/path following)
                if self.safety_alignment:
                    self.align_to_next_checkpoint()

                return
            
        return

        

# -----------------------------
# MAIN
# -----------------------------
def main():
    rclpy.init()
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
