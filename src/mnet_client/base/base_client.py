# This is the base class for the client construction
# DO NOT MODIFY THIS FILE
# Contact: support@manipulation-net.org

import re
import os
import time
import json
import base64
import logging
import threading
import subprocess
from datetime import datetime
from abc import ABC, abstractmethod

try:
    import cv2
    import rospy
    import rospkg
    import rosgraph
    import numpy as np
    from tqdm import tqdm
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import CameraInfo
    from std_srvs.srv import Trigger, TriggerResponse

except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are installed and properly configured.")
    exit()


PACKAGE_NAME = "mnet_client"
SERVER_IP = "3.21.8.9"
SERVER_PORT = 50716
AVAILABLE_TASKS = ["peg_in_hole", "block_arrangement", "grasping_in_clutter", "tabletop_manipulation", "cable_management"]
INSTRUCTION_ENABLED_TASKS = ["block_arrangement", "grasping_in_clutter", "tabletop_manipulation", "cable_management"]
OVERLAY_ENABLED_TASKS = ["grasping_in_clutter", "tabletop_manipulation"]
AUTONOMOUS_ONLY_TASKS = ["block_arrangement", "tabletop_manipulation"]
APRILTAG_ENABLED_TASKS = OVERLAY_ENABLED_TASKS

ROS_TOPIC_LANGUAGE_INSTRUCTION = "/mnet_client/current_language_instruction"
ROS_TOPIC_VISION_INSTRUCTION = "/mnet_client/current_vision_instruction"
ROS_TOPIC_TASK_SKIPPED = "/mnet_client/current_task_skipped"
ROS_TOPIC_TASK_FINISHED = "/mnet_client/current_task_finished"
ROS_TOPIC_DISCRETE_ASSISTANCE = "/mnet_client/discrete_assistance_update"
ROS_TOPIC_CONTINUOUS_ASSISTANCE = "/mnet_client/continuous_assistance_update"
ROS_TOPIC_ONGOING_TASK = "/mnet_client/ongoing_task"
ROS_TOPIC_CONNECTION_STATUS = "/mnet_client/connection_status"


def get_package_path(package_name):
    """
    Get the absolute path of a ROS package
    """
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        rospy.loginfo(f"Package path for {package_name}: {package_path}")
        return package_path
    except rospkg.ResourceNotFound as e:
        rospy.logerr(f"Package {package_name} not found: {e}")
        return None
    except Exception as e:
        rospy.logerr(f"Error getting package path: {e}")
        return None


class BaseClient(ABC):
    def __init__(self, name: str):
        """
        Local test client for the manipulation-net benchmark: manipulation-net.org
        """
        build_info = cv2.getBuildInformation()
        print(build_info)
        for line in build_info.split("\n"):
            if "FFMPEG:" in line:
                if "YES" in line:
                    pass
                else:
                    print(
                        "FFMPEG is not supported, please install ffmpeg and recompile OpenCV"
                    )
                    exit()

        rospy.init_node(name, anonymous=True)
        self.is_recording = False
        self.client_socket = None
        self.package_name = PACKAGE_NAME

        try:
            # Get the package path
            self.package_path = get_package_path(PACKAGE_NAME)
            rospy.loginfo(f"Package path: {self.package_path}")

            # Build config file path
            self.config_file = os.path.join(
                self.package_path, "config", "team_config.json"
            )
            self.banner_file = os.path.join(self.package_path, "config", "banner.txt")
            # Check if file exists
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    self.team_config = json.load(f)
                rospy.loginfo(f"Loaded config from: {self.config_file}")
            else:
                rospy.logerr(f"Config file not found: {self.config_file}")
            os.system("clear")  # Clear the terminal

            # Check if banner exists
            if os.path.exists(self.banner_file):
                with open(self.banner_file, "r") as f:
                    self.banner = f.read()
                print(self.banner)
            else:
                rospy.logerr(f"Banner loading error: {self.banner_file}")

        except Exception as e:
            rospy.logerr(f"Failed to get package path: {e}")
            exit()

        # Initialize submission details
        self.team_unique_code = self.team_config["team_unique_code"]
        self.camera_topic = self.team_config["camera_image_topic"]
        self.camera_info_topic = self.team_config["camera_info_topic"]
        self.anonymous_submission = self.team_config["anonymous_submission"]
        self.nvenc_enabled = False
        self.camera_info_loaded = False

        if not self.topic_has_publishers():
            rospy.logerr(f"Camera topic {self.camera_topic} has no publishers")
            exit()
        # Check if ffmpeg supports libx264 encoder
        if not self.check_ffmpeg_encoder("libx264"):
            rospy.logerr(
                "X264 encoder is not supported, please ensure FFmpeg compiled with libx264"
            )
            exit()

        self.bridge = CvBridge()  # Initialize CV bridge
        self.buffer_frame = None  # Buffer frame for video recording

        self.calibrated_fps = None
        self.autonomy_level = self.team_config[
            "autonomy_level"
        ]  # 0: teleoperation, 1: human-in-the-loop, 2: autonomous
        if self.autonomy_level not in [0, 1, 2]:
            rospy.logerr(
                "Invalid autonomy level, please set autonomy level to 0 (teleoperation), 1 (human-in-the-loop), or 2 (autonomous)"
            )
            exit()
        self.file_dir = os.path.join(
            self.team_config["file_dir"], datetime.now().strftime("%Y-%m-%d_%H%M%S")
        )

        self.log_file_path = None
        self.video_path = None
        self.video_path_compressed = None
        os.makedirs(self.file_dir, exist_ok=False)  # no travel through time
        self.logger = None
        self.setup_logging()

        # Subscribe to camera topic
        self.camera_sub = rospy.Subscriber(
            self.camera_topic, Image, self.camera_callback
        )
        self.camera_verified = False
        self.calculate_camera_fps()
        if self.calibrated_fps is None or self.calibrated_fps < 12:
            rospy.logerr(
                f"Camera FPS is too low (minimum 25, current: {self.calibrated_fps}), please improve your camera setup."
            )
            exit()

        # Get camera info
        self.cam_K = None
        self.cam_width = None
        self.cam_height = None
        self.det = None
        self.tag_id = None
        self.corners = None
        self.R_cw_cv = None
        self.t_cw_cv = None
        self.scene_render = None

        try:
            self.cam_K, self.cam_width, self.cam_height = self.get_camera_info()
            self.camera_info_loaded = True
        except TimeoutError as e:
            rospy.logerr(
                f"{e}, this could affect the execution of the task: {OVERLAY_ENABLED_TASKS}"
            )
        except AssertionError as e:
            rospy.logerr(f"Camera info does not match the image size: {e}")
            exit()

        # Initialize intruction status:
        self.code_writing_awaiting = True
        self.detail_demonstration_awaiting = True
        self.heartbeat_interval = 10.0  # connection check every 10 seconds

        # Initialize video writer
        self.video_writer = None
        self.start_time = None
        self.end_time = None
        self._lock = threading.Lock()

        self.current_language_instruction = None
        self.current_vision_instruction = None
        self.vision_instruction_overlay = (
            False  # if overlay the vision instruction with the camera image
        )

    def topic_has_publishers(self) -> bool:
        """
        Check if topic exists AND has publishers
        """
        master = rosgraph.Master("/check_publishers")
        state = master.getSystemState()
        publishers = state[0]

        for topic, nodes in publishers:
            if topic == self.camera_topic:
                return len(nodes) > 0
        return False

    def get_camera_info(self, timeout: float = 2.0):
        """
        Get the camera intrinsic matrix from the camera info topic
        """
        data = {"K": None, "width": None, "height": None}

        def camera_info_callback(msg: CameraInfo):
            data["K"] = np.array(msg.K).reshape(3, 3)
            data["width"] = msg.width
            data["height"] = msg.height
            rospy.loginfo(f"Received CameraInfo from {self.camera_info_topic}")

        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, camera_info_callback
        )
        end_time = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            rospy.sleep(0.1)
            if data["K"] is not None:
                break

        self.camera_info_sub.unregister()

        if data["K"] is None:
            raise TimeoutError(
                f"No CameraInfo received on {self.camera_info_topic} within {timeout}s"
            )

        W, H = (
            self.buffer_frame.shape[1],
            self.buffer_frame.shape[0],
        )  # Get the image size
        assert (
            H == data["height"] and W == data["width"]
        ), "Camera info does not match the image size"

        if data["K"] is None:
            raise TimeoutError(
                f"No CameraInfo received on {self.camera_info_topic} within {timeout}s"
            )

        return data["K"], data["width"], data["height"]

    def check_ffmpeg_encoder(self, encoder_name: str) -> bool:
        """
        Check if ffmpeg supports a specific encoder
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
            )
            if "h264_nvenc" in result.stdout:
                self.nvenc_enabled = True
            else:
                self.nvenc_enabled = False
            return encoder_name in result.stdout
        except:
            return False

    def calculate_camera_fps(self):
        """
        Calculate the camera FPS
        """
        start_time = time.time()
        count = 0
        rospy.loginfo("Calibrating frames per second, this may take a while...")

        while not self.is_recording and not rospy.is_shutdown() and count < 100:
            try:
                rospy.wait_for_message(
                    self.camera_topic, Image, timeout=1.0
                )  # waits for a frame
                count += 1
            except rospy.ROSException:
                rospy.logerr(
                    f"No image received on {self.image_topic} within 1 second — exiting."
                )
                exit()

        end_time = time.time()
        elapsed = max(1e-9, end_time - start_time)
        self.calibrated_fps = count / elapsed
        rospy.loginfo(f"Calibrated FPS: {self.calibrated_fps:.2f}")

    def compress_video(self):
        """
        Compress the video with ffmpeg
        """
        gpu_command = [
            "ffmpeg",
            "-i",
            self.video_path,
            # NVIDIA GPU encoder
            "-c:v",
            "h264_nvenc",  # Use NVIDIA hardware encoder
            "-gpu",
            "0",  # Use first GPU (if multiple GPUs)
            "-cq",
            "25",  # Quality
            "-preset",
            "p3",  # balance between quality and speed
            # Audio removal & compatibility
            "-an",  # No audio
            "-pix_fmt",
            "yuv420p",  # Universal compatibility
            "-movflags",
            "+faststart",  # Web streaming
            "-y",
            self.video_path_compressed,
        ]

        # Start ffmpeg and capture output
        process = subprocess.Popen(
            gpu_command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True
        )

        duration = None
        time_pattern = re.compile(r"time=(\d+):(\d+):(\d+).(\d+)")
        duration_pattern = re.compile(r"Duration: (\d+):(\d+):(\d+).(\d+)")

        pbar = None

        for line in process.stderr:
            if duration is None:
                match = duration_pattern.search(line)
                if match:
                    h, m, s, ms = map(int, match.groups())
                    duration = h * 3600 + m * 60 + s + ms / 100.0
                    pbar = tqdm(total=duration, unit="s", desc="Compressing")

            match = time_pattern.search(line)
            if match and pbar:
                h, m, s, ms = map(int, match.groups())
                current = h * 3600 + m * 60 + s + ms / 100.0
                pbar.update(current - pbar.n)

        if pbar:
            pbar.close()

        process.wait()
        if process.returncode == 0:
            self.logger.info("✅ Compression completed successfully.")
            return True
        else:
            self.logger.warning(
                "Compression failed. This will NOT affect your submission."
            )
            os.remove(self.video_path_compressed)
            return False

    def setup_logging(self):
        """Setup logging configuration"""

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            self.file_dir, f"submission_{self.team_unique_code}_{timestamp}.log"
        )
        self.log_file_path = log_file

        # Configure logging
        self.logger = logging.getLogger(self.package_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def print_box(self, text):
        """
        Print a box around the text
        """
        length = len(text)
        print("┌" + "─" * (length + 2) + "┐")
        print("│ " + text + " │")
        print("└" + "─" * (length + 2) + "┘")

    def decode_base64_image_rgb(self, base64_image: str):
        """
        Decode a base64 image
        """
        return cv2.imdecode(
            np.frombuffer(base64.b64decode(base64_image), np.uint8), cv2.IMREAD_COLOR
        )

    def decode_base64_image_rgba(self, base64_image: str):
        """
        Decode a base64 image
        """
        return cv2.imdecode(
            np.frombuffer(base64.b64decode(base64_image), np.uint8),
            cv2.IMREAD_UNCHANGED,
        )

    def parse_instruction(self, instruction):
        """
        Parse the instruction
        """
        self.current_language_instruction = (
            instruction.language if instruction.language else ""
        )
        if not self.vision_instruction_overlay:
            self.current_vision_instruction = (
                self.decode_base64_image_rgb(instruction.vision)
                if instruction.vision
                else None
            )
        else:
            self.current_vision_instruction = (
                self.decode_base64_image_rgba(instruction.vision)
                if instruction.vision
                else None
            )

    def add_timestamp_to_image(self, cv_image):
        """
        Add timestamp to the image
        """
        # Update the buffer frame
        elapsed = datetime.now() - self.start_time
        timestamp_text = f"{elapsed.total_seconds():.2f} s"
        cv2.putText(
            cv_image,
            timestamp_text,
            (10, 25),  # position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # font scale
            (0, 255, 0),  # text color (BGR)
            1,  # thickness
            cv2.LINE_AA,
        )
        return cv_image

    def render_overlay_image_based_on_scene_id(self, scene_id):
        if self.scene_render is not None:
            self.scene_render.load_scene(scene_id)
            rendered_scene = self.scene_render.render_scene_image()
            rendered_scene_with_axis = self.scene_render.draw_apriltag_frame(rendered_scene)
            return rendered_scene_with_axis
        else:
            return None
            
    def overlay_rgba_on_bgr(self, bg_bgr, fg_rgba):
        """
        Overlay an RGBA image (foreground) onto a BGR image (background) and return the resulting BGR image
        """
        # Resize foreground to match background
        fg_rgba = cv2.resize(
            fg_rgba, (bg_bgr.shape[1], bg_bgr.shape[0]), interpolation=cv2.INTER_AREA
        )

        fg_rgb = fg_rgba[:, :, :3].astype(float)
        alpha = fg_rgba[:, :, 3].astype(float) / 255.0
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB).astype(float)

        blended_rgb = alpha[..., None] * fg_rgb + (1 - alpha[..., None]) * bg_rgb
        blended_rgb = np.clip(blended_rgb, 0, 255).astype(np.uint8)

        blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)

        return blended_bgr

    @abstractmethod
    def camera_callback(self, msg: Image) -> None:
        """
        Callback function for the camera topic
        """
        raise NotImplementedError("Camera callback is not implemented")

    @abstractmethod
    def start_recording(self) -> bool:
        """
        Start recording the video
        """
        raise NotImplementedError("Function Start Recording is not implemented")

    @abstractmethod
    def stop_recording(self) -> bool:
        """
        Stop recording the video
        """
        raise NotImplementedError("Function Stop Recording is not implemented")

    @abstractmethod
    def run(self):
        """
        Main function to run the client
        """
        raise NotImplementedError("Function Run is not implemented")

    @abstractmethod
    def handle_task_finished(self, request) -> TriggerResponse:
        """
        Handle the service request to mark current task as finished
        """
        raise NotImplementedError("Function Handle Task Finished is not implemented")

    @abstractmethod
    def handle_task_skipped(self, request) -> TriggerResponse:
        """
        Handle the service request to mark current task as skipped
        """
        raise NotImplementedError("Function Handle Task Skipped is not implemented")

    def send_message(self, message: str) -> None:
        """
        Send a message to the server
        """
        raise NotImplementedError("Function Send Message is not implemented")

    def receive_message(self) -> str:
        """
        Receive a message from the server
        """
        raise NotImplementedError("Function Receive Message is not implemented")

    def handle_discrete_assistance(self, request) -> TriggerResponse:
        """
        Handle the service request to update the discrete assistance
        """
        raise NotImplementedError(
            "Function Handle Discrete Assistance is not implemented"
        )

    def handle_continuous_assistance(self, request) -> TriggerResponse:
        """
        Handle the service request to update the continuous assistance
        """
        raise NotImplementedError(
            "Function Handle Continuous Assistance is not implemented"
        )

    def task_execution_monitor_thread(self):
        """
        Thread function to continuously monitor connection status
        """
        raise NotImplementedError("Task Execution Status Monitor is not implemented")

    def connection_monitor_thread(self):
        """
        Thread function to periodically check connection status
        """
        raise NotImplementedError("Connection Monitor Thread is not implemented")

    def task_instruction_thread(self):
        """
        Thread function to publish vision and language instructions
        """
        raise NotImplementedError("Task Instruction Thread is not implemented")

    @abstractmethod
    def __del__(self):
        """
        Destructor to clean up resources
        """
        raise NotImplementedError("Function Destructor is not implemented")
