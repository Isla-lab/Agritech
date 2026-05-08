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
import cv2
import joblib
import random


class FiniteStateMachine(Node):
    def __init__(self):
        super().__init__("limo_yolo")

        # -----------------------------
        # ROS PARAMETERS
        # -----------------------------
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("turn_speed_max", 1.5)           # Maximum angular velocity

        self.img_topic = self.get_parameter("image_topic").value
        self.turn_speed_max = self.get_parameter("turn_speed_max").value

        # -----------------------------
        # CAMERA AND IMAGE PROCESSING
        # -----------------------------
        self.image_w = None         # Image width
        self.bridge = CvBridge()

        # -----------------------------
        # MACHINE LEARNING MODEL
        # -----------------------------
        self.model = None           # SVM Model placeholder

        # -----------------------------
        # MOTION CONTROL
        # -----------------------------
        self.rate_hz = 20
        self.forward_speed = 0.4        # Linear velocity (m/s)
        self.align_tol = 0.05           # Error tolerance for alignment phase (radians)

        # -----------------------------
        # FINITE STATE MACHINE (FSM)
        # -----------------------------
        self.state = "FORWARD"          # Current internal state of the FSM

        # -----------------------------
        # ODOMETRY DATA
        # -----------------------------
        # Current odometry pose
        self.x = None
        self.y = None
        self.yaw = None

        # Starting pose (initial reference)
        self.starting_x = None
        self.starting_y = None
        self.starting_yaw = None

        # -----------------------------
        # SEARCH AND SCAN LOGIC
        # -----------------------------
        self.search_straight_distance = 1  # Distance to travel in "FORWARD" mode before scanning

        # Starting odometry for the current search segment
        self.search_start_x = None
        self.search_start_y = None
        self.search_start_yaw = None

        # State variables for the "SCAN" phase sequence
        self.turning_right = True
        self.turning_left = False
        self.realigning = False

        self.right_target_yaw = None
        self.left_target_yaw = None
        self.realigning_target_yaw = None

        # Detection flags for symptomatic plants
        self.ill_plant_detected = False
        self.ill_plant_detected_left = False
        self.ill_plant_detected_right = False

        # -----------------------------
        # OBSTACLE AVOIDANCE
        # -----------------------------
        self.ranges = []                # LiDAR range data array

        self.obstacle_detected = False
        self.avoiding_dir = 1           # Default avoidance direction: Left
        self.avoiding = False
        self.aligning = False

        self.obstacle_threshold = 0.4   # Safety distance to trigger avoidance (meters)
        self.current_distance = np.inf  # Current distance to the nearest obstacle

        # Starting heading when entering avoidance mode
        self.start_avoiding_yaw = None

        # Mission control logic
        self.stopped = False

        # -----------------------------
        # CHECKPOINT (JUNCTION) VARIABLES
        # -----------------------------
        self.current_checkpoint = None

        # Variables for image capture and comparison at checkpoints
        self.rotated_at_checkpoint = False
        self.frontal_image = None
        self.lateral_image = None
        self.count_frontal_image = None
        self.count_lateral_image = None

        self.checkpoint_initial_yaw = None

        # Checkpoint progress tracking flags
        self.checkpoint1_reached = False
        self.checkpoint1_finished = False

        self.checkpoint2_reached = False
        self.checkpoint2_finished = False

        self.checkpoint3_reached = False
        self.checkpoint3_finished = False

        self.checkpoint4_reached = False
        self.checkpoint4_finished = False

        self.checkpoint5_reached = False
        self.checkpoint5_finished = False

        self.checkpoint6_reached = False

        # -----------------------------
        # CHECKPOINT COORDINATES (Waypoints)
        # -----------------------------
        self.checkpoint1_x = 0.525
        self.checkpoint1_y = -2.000
        self.checkpoint2_x = 0.525
        self.checkpoint2_y = 0.525
        self.checkpoint3_x = 0.525
        self.checkpoint3_y = 1.875
        self.checkpoint4_x = -1.975
        self.checkpoint4_y = 1.875
        self.checkpoint5_x = -1.975
        self.checkpoint5_y = 0.500

        # Final goal / terminal position
        self.checkpoint6_x = -1.975
        self.checkpoint6_y = -2.000

        # -----------------------------
        # ROS I/O (Subscribers and Publishers)
        # -----------------------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos)
        self.sub_image = self.create_subscription(Image, self.img_topic, self.on_image, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_image = self.create_publisher(Image, "/yolo/annotated_image", 10)

        # -----------------------------
        # MAIN CONTROL TIMER
        # -----------------------------
        self.control_timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

        # Load resources and confirm initialization
        self.load_model()
        self.get_logger().info(f"Node fully loaded and operational.")

    # -----------------------------
    # SVM MODEL LOADING
    # -----------------------------
    def load_model(self):
        """
        Loads the SVM model and class labels from joblib files.

        Initializes:
            self.model: The loaded SVM classifier.
            self.class_names: The categorical labels associated with the model.
        """

        # Expand the '~' symbol or automatically retrieve the user's home directory
        home = os.path.expanduser("~")
        model_path = os.path.join(home, "Agritech/workspace/assets/leaf_svm_model.joblib")
        labels_path = os.path.join(home, "Agritech/workspace/assets/leaf_labels.joblib")

        t0 = time.time()
        # Deserialize the model and labels from disk
        self.model = joblib.load(model_path)
        self.class_names = joblib.load(labels_path)
        
        self.get_logger().info(f"Loaded SVM model in {time.time() - t0:.2f}s")

    # -----------------------------
    # ODOMETRY CALLBACK
    # -----------------------------
    def odom_callback(self, msg: Odometry):
        """
        Updates the robot's pose based on odometry data.

        Extracts X and Y positions and converts the orientation quaternion 
        into a yaw angle (Euler).
        """
        # Update current spatial coordinates
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Extract orientation as a quaternion
        q = msg.pose.pose.orientation
        
        # Convert to 2D heading (yaw)
        self.yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # Initialize starting pose on the first received message
        if self.starting_x is None:
            self.starting_x = self.x
            self.starting_y = self.y
            self.starting_yaw = self.yaw
            self.get_logger().info("Odom initialized")

    # -----------------------------
    # CAMERA CALLBACK
    # -----------------------------
    def on_image(self, msg: Image):
        """
        Callback for image processing.

        Converts the ROS Image message to OpenCV format, manages frame capturing 
        at checkpoints for comparison, and analyzes the "light green" pixel 
        ratio to detect potentially diseased plants.

        Args:
            msg (Image): Image message received from the camera sensor.
        """

        # Convert ROS Image message to OpenCV BGR format
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return
        
        # Capture and process frontal/lateral frames for CHECKPOINT_1
        if self.checkpoint1_reached and self.frontal_image is None and not self.checkpoint1_finished:
            self.frontal_image = img_bgr.copy()
            self.count_frontal_image = self.count_ill_pixels(self.frontal_image)
            return

        if self.rotated_at_checkpoint and self.lateral_image is None and not self.checkpoint1_finished:
            self.lateral_image = img_bgr.copy()
            self.count_lateral_image = self.count_ill_pixels(self.lateral_image)
            return
        
        # Capture and process frontal/lateral frames for CHECKPOINT_2
        if self.checkpoint2_reached and self.frontal_image is None and not self.checkpoint2_finished:
            self.frontal_image = img_bgr.copy()
            self.count_frontal_image = self.count_ill_pixels(self.frontal_image)
            return

        if self.rotated_at_checkpoint and self.lateral_image is None and not self.checkpoint2_finished:
            self.lateral_image = img_bgr.copy()
            self.count_lateral_image = self.count_ill_pixels(self.lateral_image)
            return
        
        # Capture and process frontal/lateral frames for CHECKPOINT_5
        if self.checkpoint5_reached and self.frontal_image is None and not self.checkpoint5_finished:
            self.frontal_image = img_bgr.copy()
            self.count_frontal_image = self.count_ill_pixels(self.frontal_image)
            return

        if self.rotated_at_checkpoint and self.lateral_image is None and not self.checkpoint5_finished:
            self.lateral_image = img_bgr.copy()
            self.count_lateral_image = self.count_ill_pixels(self.lateral_image)
            return

        # Extract BGR channels for color segmentation
        blue_channel = img_bgr[:, :, 0]
        green_channel = img_bgr[:, :, 1]
        red_channel = img_bgr[:, :, 2]

        # Apply thresholding to identify "light green" (potential plant disease)
        condition = (green_channel > 80) & (green_channel < 190) & \
                    (red_channel < 110) & (red_channel > 50) & \
                    (blue_channel < 70)

        light_green_count = np.count_nonzero(condition)

        # Calculate the percentage of symptomatic pixels in the current frame
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        percentage = (light_green_count / total_pixels) * 100

        # Trigger warnings and update scan state flags if detection exceeds threshold
        if percentage > 3.0:
            self.get_logger().warn(f"WARNING: Light green detection high! Ratio: {percentage:.2f}%")

            if self.state == "SCAN":
                # Record which side of the scan detected the issue
                if self.turning_right:
                    self.ill_plant_detected_right = True
                
                if self.turning_left:
                    self.ill_plant_detected_left = True

        h, w = img_bgr.shape[:2]

        if self.image_w is None:
            self.image_w = w

        return


    # -----------------------------
    # LIDAR CALLBACK
    # -----------------------------
    def on_scan(self, msg: LaserScan):
        """
        Callback for LiDAR data. Updates the minimum frontal distance 
        by analyzing a 40-sample central sector of the laser scan.
        """

        self.ranges = np.array(msg.ranges)

        # Focus on the frontal field of view (FOV)
        # Slices a 40-point window around the center of the laser array
        n = len(self.ranges)
        center = n // 2
        left = max(center - 20, 0)
        right = min(center + 20, n)
        
        if left < right:
            # np.nanmin ignores NaN values (common in LiDAR if the beam doesn't return)
            self.current_distance = np.nanmin(self.ranges[left:right])
        else:
            self.current_distance = np.nanmin(self.ranges)

    # -----------------------------
    # CONTROL METHODS
    # -----------------------------
    def count_ill_pixels(self, img_bgr):
        """
        Counts pixels matching the "symptomatic green" color range 
        using BGR channel filtering.

        Returns:
            int: Number of detected pixels.
        """
        
        # Extract BGR channels
        blue_channel = img_bgr[:, :, 0]
        green_channel = img_bgr[:, :, 1]
        red_channel = img_bgr[:, :, 2]

        # Condition to identify symptomatic green foliage
        condition = (green_channel > 80) & (green_channel < 190) & \
                    (red_channel < 110) & (red_channel > 50) & \
                    (blue_channel < 70)

        # Count how many pixels in the image satisfy the condition
        light_green_count = np.count_nonzero(condition)

        return light_green_count

    def distance_from_point(self, x0: float, y0: float) -> float:
        """
        Calculates the Euclidean distance from the current position (self.x, self.y)
        to a specified point (x0, y0).
        """
        # math.hypot calculates the square root of the sum of squares
        return math.hypot(self.x - x0, self.y - y0)
    
    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Calculates the yaw (rotation around Z-axis) from a quaternion.
        """
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy*qy + qz*qz)
        return math.atan2(siny, cosy)

    def normalize_angle(self, angle: float) -> float:
        """
        Normalizes an angle to the range [-pi, pi).
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def angle_error(self, target_angle: float) -> float:
        """
        Calculates the normalized angular error between the target angle and current yaw.
        """
        return self.normalize_angle(target_angle - self.yaw)

    def calculate_target_yaw(self, goal_x: float, goal_y: float) -> float:
        """
        Calculates the yaw angle required to point the robot toward the goal.
        """
        return math.atan2(goal_y - self.y, goal_x - self.x)
    
    def publish_twist(self, linear=0.0, angular=0.0):
        """
        Publishes linear and angular velocity commands to the /cmd_vel topic.

        Args:
            linear (float): Linear velocity in X (m/s).
            angular (float): Angular velocity in Z (rad/s).
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        """Publishes a zero-velocity command to stop the robot."""
        self.publish_twist(0.0, 0.0)

    def print_message(self, message):
        """Logs an info-level message to the console."""
        self.get_logger().info(message)
        

    # -----------------------------
    # OBSTACLE DETECTION
    # -----------------------------
    def detect_obstacle(self):
        """
        Checks for the presence of obstacles.

        Prioritizes safety: an obstacle is flagged if the frontal LiDAR distance 
        is below the safety threshold.
        """
 
        # Detects a generic obstacle if the frontal LiDAR reading is too close
        self.obstacle_detected = self.current_distance < self.obstacle_threshold


    def align_to_next_checkpoint(self):
        """
        Calculates and rotates the robot toward the next waypoint.

        Determines the target goal based on the current checkpoint count, 
        calculates the required heading (yaw), and applies proportional 
        control to rotate the robot until it is within the alignment tolerance.
        """
        goal_x = None
        goal_y = None

        # Determine the position of the checkpoint to align with 
        # (the one following the most recently reached checkpoint)
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
        elif self.current_checkpoint == 4:
            goal_x = self.checkpoint5_x
            goal_y = self.checkpoint5_y
        elif self.current_checkpoint == 5:
            goal_x = self.checkpoint6_x
            goal_y = self.checkpoint6_y

        # Calculate the required heading to face the goal
        target_yaw = self.calculate_target_yaw(goal_x, goal_y)
        angle_err = self.angle_error(target_yaw)
            
        if abs(angle_err) > self.align_tol:
            # Rotate using proportional control (P-controller)
            # The speed is limited by turn_speed_max to ensure smooth motion
            angular_vel = max(min(0.8 * angle_err, self.turn_speed_max), -self.turn_speed_max)
            self.publish_twist(0.0, angular_vel)
        else:
            # Alignment complete; stop rotating
            self.aligning = False

        return

    def check_for_checkpoint_reached(self):
        """
        Monitors the robot's proximity to predefined coordinates (checkpoints).
        
        When a checkpoint is reached (distance <= 0.1m), it triggers the 
        corresponding state or flag. Some checkpoints trigger an inspection 
        state (e.g., CHECKPOINT_1, 2, 5), while others simply act as 
        waypoint counters or final stop triggers.
        """

        # CHECKPOINT_1: Triggers inspection logic
        if not self.checkpoint1_reached and not self.checkpoint1_finished:
            d = self.distance_from_point(self.checkpoint1_x, self.checkpoint1_y)
            if d <= 0.1:
                self.state = "CHECKPOINT_1"
                self.checkpoint1_reached = True

                if self.checkpoint_initial_yaw is None:
                    # Store heading for reference during rotation
                    self.checkpoint_initial_yaw = self.yaw
                    
                self.publish_stop()
                return 
            
        # CHECKPOINT_2: Triggers inspection logic
        if not self.checkpoint2_reached and not self.checkpoint2_finished:
            d = self.distance_from_point(self.checkpoint2_x, self.checkpoint2_y)
            if d <= 0.1:
                self.state = "CHECKPOINT_2"
                self.checkpoint2_reached = True

                if self.checkpoint_initial_yaw is None:
                    self.checkpoint_initial_yaw = self.yaw

                self.publish_stop()
                return 
            
        # CHECKPOINT_3: Passive waypoint (updates counter only)
        if not self.checkpoint3_reached and not self.checkpoint3_finished:
            d = self.distance_from_point(self.checkpoint3_x, self.checkpoint3_y)
            if d <= 0.1:
                self.checkpoint3_reached = True
                self.checkpoint3_finished = True
                self.current_checkpoint += 1
                return      

        # CHECKPOINT_4: Passive waypoint (updates counter only)
        if not self.checkpoint4_reached and not self.checkpoint4_finished:
            d = self.distance_from_point(self.checkpoint4_x, self.checkpoint4_y)
            if d <= 0.1:
                self.checkpoint4_reached = True
                self.checkpoint4_finished = True
                self.current_checkpoint += 1
                return       
            
        # CHECKPOINT_5: Triggers inspection logic
        if not self.checkpoint5_reached and not self.checkpoint5_finished:
            d = self.distance_from_point(self.checkpoint5_x, self.checkpoint5_y)
            if d <= 0.1:
                self.state = "CHECKPOINT_5"
                self.checkpoint5_reached = True

                if self.checkpoint_initial_yaw is None:
                    self.checkpoint_initial_yaw = self.yaw

                self.publish_stop()
                return 
            
        # CHECKPOINT_6: Final destination
        if not self.checkpoint6_reached:
            d = self.distance_from_point(self.checkpoint6_x, self.checkpoint6_y)
            if d <= 0.1:
                # Permanently stops the robot upon reaching the end goal
                self.stopped = True     
                self.publish_stop()
                return


    # -----------------------------
    # STATE METHODS
    # -----------------------------
    def move_forward(self):
        """
        Manages the linear advancement phase during search mode.
        
        The method performs the following operations:
        1. Verifies that the current state is indeed "FORWARD".
        2. Stores the initial position (odometry) at the first start of the maneuver.
        3. Calculates the distance traveled relative to the starting point.
        4. If the traveled distance is equal to or greater than 'search_straight_distance':
           - Stops the robot.
           - Resets all navigation and orientation parameters.
           - Transitions to the "SCAN" state to begin the search rotation.
        5. If the distance has not yet been reached, continues publishing
           a constant linear velocity command.
        """
        
        # Safety check: if the internal state is not FORWARD -> exit immediately
        if self.state != "FORWARD": 
            return
        
        # Rotate toward the next checkpoint (junction) before moving
        if self.aligning:
            self.align_to_next_checkpoint()
            return

        # Save starting odometry coordinates
        if self.search_start_x is None:
            self.search_start_x = self.x
            self.search_start_y = self.y
            self.search_start_yaw = self.yaw

        # Calculate distance from the start point and check against target
        d = self.distance_from_point(self.search_start_x, self.search_start_y)

        if d >= self.search_straight_distance:
            self.publish_stop()

            # Initialize scan parameters for the next state
            self.turning_right = True
            self.turning_left = False
            self.realigning = False

            self.right_target_yaw = None
            self.left_target_yaw = None
            self.realigning_target_yaw = None

            # Reset local state variables
            self.search_start_x = None
            self.search_start_y = None
            self.search_start_yaw = None

            # Transition to the SCAN state
            self.state = "SCAN"

            return

        # Otherwise, continue moving straight
        self.publish_twist(self.forward_speed, 0.0)

        return

    def scan(self):
        """
        Executes an angular scanning maneuver in place to search for a target.
        
        The logic follows a three-phase sequence:
        1. RIGHT ROTATION: Rotates the robot 45 degrees to the right relative 
           to the initial heading.
        2. LEFT ROTATION: Once the right turn is complete, rotates to 
           45 degrees left relative to the starting heading (90° total arc).
        3. REALIGNMENT: Returns to the original heading stored at the start 
           of the scan.
        
        Upon completion of realignment, the state is set to "FORWARD" to 
        continue straight exploration if no symptomatic plants were identified; 
        otherwise, the state is set to "ANALYZE" to process a sample of the 
        detected plant.
        """
        # Safety check: if the internal state is not SCAN -> exit immediately
        if self.state != "SCAN":
            return

        # Phase 1: Turn Right
        if self.turning_right:

            if self.right_target_yaw is None:
                # Save initial heading for subsequent realignment
                self.start_spinning_yaw = self.yaw  
                self.right_target_yaw = self.yaw - math.pi / 4

            # If target amplitude is reached, stop rotating
            if abs(self.angle_error(self.right_target_yaw)) < self.align_tol:
                self.publish_stop()

                # Update internal state flags
                self.right_target_yaw = None
                self.turning_left = True
                self.turning_right = False

                return
            
            # Otherwise, continue rotating RIGHT
            self.publish_twist(0.0, -0.5)

            return
        
        # Phase 2: Turn Left
        if self.turning_left:

            if self.left_target_yaw is None:
                # Target is 45 degrees left of the original STARTING heading
                self.left_target_yaw = self.start_spinning_yaw + math.pi / 4

            # If target amplitude is reached, stop rotating
            if abs(self.angle_error(self.left_target_yaw)) < 0.05:
                self.publish_stop()
                self.left_target_yaw = None
                self.realigning = True
                self.turning_left = False

                return
            
            # Otherwise, continue rotating LEFT
            self.publish_twist(0.0, 0.5)
            return
        
        # Phase 3: Realign to starting heading
        if self.realigning:
            
            if self.realigning_target_yaw is None:
                self.realigning_target_yaw = self.start_spinning_yaw

            # If realignment complete: stop rotating and decide next state
            if abs(self.angle_error(self.realigning_target_yaw)) < self.align_tol:
                self.publish_stop()
                self.realigning_target_yaw = None
                self.realigning = False

                # Robot is now correctly oriented.
                # If symptomatic plants were detected during the scan arc, analyze a sample
                if self.ill_plant_detected_left or self.ill_plant_detected_right:
                    self.ill_plant_detected_left = False
                    self.ill_plant_detected_right = False

                    self.aligning = True
                    self.state = "ANALYZE"

                    return

                # Otherwise, proceed straight
                # Reset FORWARD state variables for fresh odometry tracking
                self.search_start_x = None
                self.search_start_y = None
                self.search_start_yaw = None

                self.aligning = True
                self.state = "FORWARD"

                return
            
            # Otherwise, rotate back toward the center (right)
            self.publish_twist(0.0, -0.5)
            return

        return

    def avoid(self):
        """
        Simplified obstacle avoidance adapted to the mission context.

        The robot checks alternative directions sequentially (Left, Right, Backwards) 
        until a clear path is found. Once the path is clear, it resets the 
        navigation variables and returns to the FORWARD state.
        """

        # Initialization
        if self.start_avoiding_yaw is None:
            self.start_avoiding_yaw = self.yaw
            self.avoid_step = 0 # 0: Look Left, 1: Look Right, 2: Look Backwards
            self.avoid_target_yaw = None
            return

        # Define the rotation angle for the current step
        if self.avoid_target_yaw is None:
            if self.avoid_step == 0:
                target_offset = math.pi / 2   # +90° (Left)
            elif self.avoid_step == 1:
                target_offset = -math.pi / 2  # -90° (Right)
            else:
                target_offset = -math.pi      # -180° (Backwards)
            
            self.avoid_target_yaw = self.normalize_angle(self.start_avoiding_yaw + target_offset)

        # Execute Rotation
        angle_error = self.angle_error(self.avoid_target_yaw)

        if abs(angle_error) > 0.1:
            # Determine shortest rotation direction
            direction = 1.0 if angle_error > 0 else -1.0
            self.publish_twist(0.0, direction * 0.5) 
            return

        # Pause to check the path in the new direction
        self.publish_stop()
        
        if not self.obstacle_detected:
            # Path is clear: Resume navigation
            self.start_avoiding_yaw = None
            self.avoid_target_yaw = None

            # Reset odometry tracking variables for FORWARD state
            self.search_start_x = self.search_start_y = self.search_start_yaw = None
            self.aligning = True
            self.state = "FORWARD"
            return
        else:
            # Path still blocked: Increment step to try another direction
            self.avoid_step += 1
            self.avoid_target_yaw = None
            
            # If all directions are blocked, stop the robot entirely
            if self.avoid_step > 2:
                self.stopped = True


    def analyze(self):
        """
        Performs plant analysis using SVM classification.

        Randomly selects an image from the data directory (70% Blight, 30% Healthy),
        extracts HOG features, and uses the SVM model for inference.
        Publishes the visual result and resets the state to FORWARD.
        """
        self.publish_stop()

        # Path configuration and parameters
        IMG_SIZE = (128, 128)
        home = os.path.expanduser("~")
        base_path = os.path.join(home, "Agritech/workspace/Data/Original Data/")

        # 1. Select folder with weighted probability
        # Simulates real-world distribution: 70% Blight, 30% Healthy
        subfolder = random.choices(
            ["Leaf Blight", "Healthy"], 
            weights=[0.7, 0.3], 
            k=1
        )[0]
        
        folder_path = os.path.join(base_path, subfolder)

        # 2. Select a random image from the chosen folder
        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                raise FileNotFoundError(f"No image files found in {folder_path}")
            
            img_path = os.path.join(folder_path, random.choice(files))
            img = cv2.imread(img_path)
            
            if img is None:
                raise ValueError(f"Unable to read image: {img_path}")

        except Exception as e:
            self.get_logger().error(f"Image loading error: {e}")

            # Reset state variables and resume navigation
            self.search_start_x = self.search_start_y = self.search_start_yaw = None
            self.aligning = True
            self.state = "FORWARD"
            return

        # Image processing
        display_img = img.copy() 
        img_resized = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # HOG + SVM Inference
        # Histogram of Oriented Gradients (HOG) captures the structural shape of the leaf
        hog = cv2.HOGDescriptor((128,128), (32,32), (16,16), (16,16), 9)
        features = hog.compute(gray).flatten().reshape(1, -1)
        
        pred_idx = self.model.predict(features)[0]
        prob = self.model.predict_proba(features).max()
        label = self.class_names[pred_idx]

        # Graphical overlay of results
        text = f"{label} ({prob*100:.1f}%)"
        cv2.putText(display_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Publish the annotated image
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        try:
            # Convert OpenCV image to ROS message using "rgb8" encoding
            annotated_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            self.pub_image.publish(annotated_msg)
            self.get_logger().info(f"Analysis completed: {text}")
        except Exception as e:
            self.get_logger().error(f"Image publication error: {e}")

        # Reset state variables and resume navigation
        self.search_start_x = self.search_start_y = self.search_start_yaw = None
        self.aligning = True
        self.state = "FORWARD"


    def checkpoint(self, direction, check_num):
        """
        Manages checkpoint inspection logic via rotation and visual comparison.

        Performs a 90° rotation to acquire a lateral image, compares the 
        number of "symptomatic" pixels between the frontal and lateral views 
        to determine the optimal orientation, and then resets the state 
        to resume navigation.

        Args:
            direction (int): Rotation direction (1 for left, -1 for right).
            check_num (int): Numerical ID of the current checkpoint.
        """

        # Wait until the frontal image is stored
        if self.frontal_image is None:
            return
        
        # Rotate 90 degrees in the specified direction 
        angle_error = self.angle_error(self.checkpoint_initial_yaw + (direction*math.pi/2))

        if abs(angle_error) >= 0.1 and not self.rotated_at_checkpoint:
            self.publish_twist(0.0, 0.3*direction)
            print(f"Rotating: {self.rotated_at_checkpoint}")
            return
        
        if not self.rotated_at_checkpoint:
            self.rotated_at_checkpoint = True
            self.publish_stop()     # Stop rotating once aligned

        if self.lateral_image is None:
            return
        
        # Compare the two images and decide which direction to take
        if self.count_frontal_image is None:
            self.count_frontal_image = self.count_ill_pixels(self.frontal_image)
            return
        
        if self.count_lateral_image is None:
            self.count_lateral_image = self.count_ill_pixels(self.lateral_image)
            return

        print(f"FRONTAL IMAGE COUNT: {self.count_frontal_image}")
        print(f"LATERAL IMAGE COUNT: {self.count_lateral_image}")

        if self.count_frontal_image < self.count_lateral_image:
            # Optimal view found; ensure the robot stays stationary
            self.publish_stop()     

        else:
            # Re-align to the initial direction if frontal was better/equivalent
            angle_error = self.angle_error(self.checkpoint_initial_yaw)

            if abs(angle_error) >= 0.1:
                self.publish_twist(0.0, -0.3*direction)
                return

        # Update specific checkpoint flags
        if check_num == 1:
            self.checkpoint1_reached = False      
            self.checkpoint1_finished = True
        elif check_num == 2:
            self.checkpoint2_reached = False      
            self.checkpoint2_finished = True     
        elif check_num == 5:
            self.checkpoint5_reached = False
            self.checkpoint5_finished = True

        # Reset generic CHECKPOINT state variables
        self.rotated_at_checkpoint = False
        self.frontal_image = None
        self.lateral_image = None
        self.count_frontal_image = None
        self.count_lateral_image = None   
        self.checkpoint_initial_yaw = None      

        # Increment current checkpoint counter
        if self.current_checkpoint is None:
            self.current_checkpoint = 1
        else:
            self.current_checkpoint += 1 

        # Reset FORWARD state variables for a fresh start
        self.search_start_x = None
        self.search_start_y = None
        self.search_start_yaw = None

        self.aligning = True

        # Return to the FORWARD navigation state
        self.state = "FORWARD"
        return     

    # -----------------------------
    # CONTROL LOOP
    # -----------------------------
    def control_loop(self):
        """
        Main robot control loop (Finite State Machine).

        Manages state transitions based on position (checkpoints), 
        safety (obstacle avoidance), and mission logic. Coordinates 
        the execution of navigation, scanning, and analysis behaviors.
        """

        # Ensure sensors (Odom/Lidar) are initialized before proceeding
        if self.x is None or self.yaw is None or self.ranges is None:
            self.publish_stop()
            return
        
        # Check if a junction or checkpoint has been reached
        self.check_for_checkpoint_reached()

        # Obstacle avoidance logic
        self.detect_obstacle()
        if self.obstacle_detected and self.state == "FORWARD":
            self.state = "AVOID"

        # Execute the logic appropriate for the current state
        # TODO

# -----------------------------
# MAIN
# -----------------------------
def main():
    rclpy.init()
    node = FiniteStateMachine()
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
