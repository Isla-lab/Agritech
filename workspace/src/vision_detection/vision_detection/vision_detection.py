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
import joblib
import random


class VisionDetectionNode(Node):
    def __init__(self):
        super().__init__("limo_yolo")

        # ROS PARAMETERS
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("turn_speed_max", 1.5)           # Maximum angular velocity

        self.img_topic = self.get_parameter("image_topic").value
        self.turn_speed_max = self.get_parameter("turn_speed_max").value

        # CAMERA AND IMAGE
        self.image_w = None         # Image width
        self.bridge = CvBridge()

        # MODEL
        self.model = None
        self.class_names = None

        # MOVEMENT CONTROL
        self.rate_hz = 20
        self.forward_speed = 0.4        # Linear speed
        self.align_tol = 0.03           # Error tolerance for realignment

        # NAVIGATION FLAGS
        self.scanning = False
        self.analyzing = False
        self.stopped = False            # Permanent stop flag

        # ODOMETRY
        # Current odometry
        self.x = None
        self.y = None
        self.yaw = None

        # Initial odometry
        self.starting_x = None
        self.starting_y = None
        self.starting_yaw = None

        # SEARCH AND SCAN DATA
        self.search_straight_distance = 1.0 # Straight distance to travel during movement phase

        # Reference coordinates for search
        self.search_start_x = None
        self.search_start_y = None
        self.search_start_yaw = None

        # Control variables for the panoramic sweep
        self.turning_right = True
        self.turning_left = False
        self.realigning = False

        self.right_target_yaw = None
        self.left_target_yaw = None
        self.realigning_target_yaw = None
        self.start_spinning_yaw = None

        # Detection flags for plant health
        self.ill_plant_detected_left = False
        self.ill_plant_detected_right = False

        # Percentage threshold to detect ill plants
        self.color_threshold = 10   # 10%

        # Final Goal Coordinates
        self.goal_x = -1.975
        self.goal_y = -0.025

        # -----------------------------
        # ROS I/O
        # -----------------------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos)
        self.sub_image = self.create_subscription(Image, self.img_topic, self.on_image, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_image = self.create_publisher(Image, "/yolo/annotated_image", 10)

        # Main execution loop timer
        self.control_timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

        self.load_model()
        self.get_logger().info("Node initialized and SVM model loading...")

    # -----------------------------
    # MODEL LOADING
    # -----------------------------
    def load_model(self):
        """
        Loads the SVM model and class labels from joblib files.
        Accesses local assets to prepare for inference.
        """
        home = os.path.expanduser("~")
        model_path = os.path.join(home, "Agritech/workspace/assets/leaf_svm_model.joblib")
        labels_path = os.path.join(home, "Agritech/workspace/assets/leaf_labels.joblib")

        try:
            t0 = time.time()
            self.model = joblib.load(model_path)
            self.class_names = joblib.load(labels_path)
            self.get_logger().info(f"Loaded SVM model in {time.time() - t0:.2f}s")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")

    # -----------------------------
    # ODOMETRY CALLBACK
    # -----------------------------
    def odom_callback(self, msg: Odometry):
        """Updates the robot's current pose based on sensor feedback."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        if self.starting_x is None:
            self.starting_x = self.x
            self.starting_y = self.y
            self.starting_yaw = self.yaw
            self.get_logger().info("Initial odometry coordinates captured.")

    # -----------------------------
    # CAMERA CALLBACK
    # -----------------------------
    def on_image(self, msg: Image):
        """
        Processes incoming camera frames to monitor specific color ranges.
        
        During the panoramic sweep, it checks for a high ratio of "light green" pixels.
        High ratios indicate a potential 'ill' plant in the robot's current field of view.
        """
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        # Isolate color channels
        blue_channel = img_bgr[:, :, 0]
        green_channel = img_bgr[:, :, 1]
        red_channel = img_bgr[:, :, 2]

        # TODO: Filter for symptomatic leaf color ranges
        # TRY TO TUNE THE THRESHOLDS AND SEE THE DIFFERENCES!
        condition = (green_channel > 80) & (green_channel < 190) & \
                    (red_channel < 110) & (red_channel > 50) & \
                    (blue_channel < 70)

        light_green_count = np.count_nonzero(condition)
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        percentage = (light_green_count / total_pixels) * 100

        # Flag detection if the specific color ratio exceeds threshold
        if percentage > self.color_threshold:
            if self.scanning:
                if self.turning_right:
                    self.ill_plant_detected_right = True
                if self.turning_left:
                    self.ill_plant_detected_left = True

        if self.image_w is None:
            self.image_w = img_bgr.shape[1]

    # -----------------------------
    # UTILITY METHODS
    # -----------------------------
    def distance_from_point(self, x0: float, y0: float) -> float:
        """Calculates Euclidean distance to target coordinates."""
        return math.hypot(self.x - x0, self.y - y0)
    
    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """Translates orientation data into a yaw angle (radians)."""
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy*qy + qz*qz)
        return math.atan2(siny, cosy)

    def normalize_angle(self, angle: float) -> float:
        """Ensures an angle remains within the [-pi, pi) boundary."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def angle_error(self, target_angle: float) -> float:
        """Measures the difference between a target heading and current yaw."""
        return self.normalize_angle(target_angle - self.yaw)

    def calculate_target_yaw(self, goal_x: float, goal_y: float) -> float:
        """Determines the heading required to reach the final destination."""
        return math.atan2(goal_y - self.y, goal_x - self.x)
    
    def publish_twist(self, linear=0.0, angular=0.0):
        """Sends movement commands to the robot base."""
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        """Halts all robot movement."""
        self.publish_twist(0.0, 0.0)

    # -----------------------------
    # TASK METHODS (SCAN & ANALYZE)
    # -----------------------------
    def scan(self):
        """
        Executes a panoramic sweep to look for targets.
        
        Logic sequence:
        1. RIGHT SWEEP: Rotates 45 degrees right from start heading.
        2. LEFT SWEEP: Rotates across to 45 degrees left (90° total arc).
        3. REALIGNMENT: Aligns heading back towards the goal.
        
        If a target was flagged during the sweep, the robot proceeds to classification.
        """
        if not self.scanning:
            return

        # Phase 1: Rotating Right
        # TODO: Implement right sweep logic.
        # Command a rightward twist until 45 degrees is reached.
        # Upon completion, stop the robot and update the following state variables:
        # - self.right_target_yaw
        # - self.turning_left
        # - self.turning_right
        pass
        
        # Phase 2: Rotating Left
        # TODO: Implement left sweep logic.
        # Command a leftward twist until the left target is reached.
        # Upon completion, stop the robot and update the following state variables:
        # - self.left_target_yaw
        # - self.realigning
        # - self.turning_left
        pass
        
        # Phase 3: Goal Alignment
        # TODO: Implement goal alignment and post-scan evaluation.
        # Command rotation back to the original goal trajectory.
        # Upon completion, stop the robot and update the following state variables:
        # - self.realigning
        # - self.scanning
        # Finally, evaluate vision flags (self.ill_plant_detected_left, self.ill_plant_detected_right).
        # Depending on detection, update either:
        # - self.analyzing (and reset vision flags) 
        # OR
        # - self.search_start_x
        pass

    def analyze(self):
        """
        Performs target classification using vision models.
        
        1. Selects a sample image from the database.
        2. Generates HOG features for the visual sample.
        3. Runs inference via the SVM classifier.
        4. Publishes a visual report to the annotated image topic.
        """
        self.publish_stop()

        IMG_SIZE = (128, 128)
        home = os.path.expanduser("~")
        base_path = os.path.join(home, "Agritech/workspace/Data/Original Data/")

        # Random sample selection (probabilistic weighting)
        subfolder = random.choices(["Leaf Blight", "Healthy"], weights=[0.7, 0.3], k=1)[0]
        folder_path = os.path.join(base_path, subfolder)

        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files: raise FileNotFoundError("No images found.")
            
            img_path = os.path.join(folder_path, random.choice(files))
            img = cv2.imread(img_path)
            if img is None: raise ValueError("Invalid image.")
        except Exception as e:
            self.get_logger().error(f"Sample error: {e}")
            self.analyzing = False
            self.search_start_x = None
            return

        # Pre-processing and Feature Generation
        display_img = img.copy() 
        img_resized = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Compute HOG features
        hog = cv2.HOGDescriptor((128,128), (32,32), (16,16), (16,16), 9)
        features = hog.compute(gray).flatten().reshape(1, -1)
        
        # Classifier Inference
        pred_idx = self.model.predict(features)[0]
        prob = self.model.predict_proba(features).max()
        label = self.class_names[pred_idx]

        # Report Generation
        text = f"{label} ({prob*100:.1f}%)"
        cv2.putText(display_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            self.pub_image.publish(annotated_msg)
            self.get_logger().info(f"Analysis complete: {text}")
        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")

        # Transition to next movement segment
        self.search_start_x = None
        self.analyzing = False

    # -----------------------------
    # MAIN EXECUTION LOOP
    # -----------------------------
    def control_loop(self):
        """
        Main logic loop running at a fixed frequency.
        
        Coordinates the sequence of behaviors:
        1. Halt if the final goal is reached.
        2. Perform panoramic sweep if triggered.
        3. Run classification tasks if targets were found.
        4. Move straight for defined intervals when searching.
        """
        if self.x is None or self.yaw is None or self.stopped:
            self.publish_stop()
            return

        # Final destination check
        dist_to_goal = self.distance_from_point(self.goal_x, self.goal_y)
        if dist_to_goal <= 0.1:
            self.get_logger().info("Goal reached. Stopping.")
            self.publish_stop()
            self.stopped = True
            return

        # Decision Logic
        if self.scanning:
            self.scan()
        elif self.analyzing:
            self.analyze()
        else:
            # Movement segment initialization
            if self.search_start_x is None:
                self.search_start_x = self.x
                self.search_start_y = self.y

            # Evaluate distance traveled in current segment
            traveled_dist = self.distance_from_point(self.search_start_x, self.search_start_y)
            if traveled_dist >= self.search_straight_distance:
                self.publish_stop()
                
                # Trigger panoramic sweep
                self.turning_right = True
                self.turning_left = False
                self.realigning = False
                self.right_target_yaw = None
                self.left_target_yaw = None
                self.scanning = True
            else:
                # Continue forward motion
                self.publish_twist(self.forward_speed, 0.0)

# -----------------------------
# EXECUTION
# -----------------------------
def main():
    rclpy.init()
    node = VisionDetectionNode()
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
