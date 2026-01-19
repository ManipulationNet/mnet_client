# Implementation of the submission client for ManipulationNet: manipulation-net.org
# DO NOT MODIFY THIS FILE
# Contact: support@manipulation-net.org

import os
import sys
import time
import struct
import socket
import select
import hashlib
import zipfile
import threading
from datetime import datetime

try:
    import cv2
    import rospy
    import requests
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Bool
    from std_srvs.srv import Trigger, TriggerResponse
    from mnet_client.tasks import detect_apriltag
    from mnet_client.base import (
        BaseClient,
        SERVER_IP,
        SERVER_PORT,
        ServerResponse,
        HEADER_FMT,
        HEADER_SIZE,
        OVERLAY_ENABLED_TASKS,
        AUTONOMOUS_ONLY_TASKS,
        APRILTAG_ENABLED_TASKS,
        ROS_TOPIC_CONTINUOUS_ASSISTANCE,
        ROS_TOPIC_DISCRETE_ASSISTANCE,
        ROS_TOPIC_CONNECTION_STATUS,
        ROS_TOPIC_ONGOING_TASK,
        ROS_TOPIC_LANGUAGE_INSTRUCTION,
        ROS_TOPIC_VISION_INSTRUCTION,
        ROS_TOPIC_TASK_SKIPPED,
        ROS_TOPIC_TASK_FINISHED
    )
    from mnet_client.base import (
        PingRequest,
        LoginRequest,
        ShutdownRequest,
        TaskRequest,
        HashUpdateRequest,
        ExecutionStatusRequest,
        InstructionRequest,
        AssistanceRequest,
        SubmissionRequest,
        CameraConfigRequest,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are installed and properly configured.")
    exit()


class SubmissionClient(BaseClient):
    """
    The main class for the ManipulationNet client
    """

    def __init__(self):
        super().__init__("submission_client")

        # Initialize submission details
        self.server_port = SERVER_PORT
        self.server_ip = SERVER_IP
        self.one_time_code = None

        # Initialize task details
        self.scoring_details = None
        self.scoring_details_list = None
        self.finished_tasks = []
        self.collected_points = 0
        self.current_task_id = 0
        self.instruction_enabled = False
        self.current_language_instruction = None
        self.current_vision_instruction = None
        self.total_tasks_number = None
        self.continuous_assistance_enabled = False

        # Recording related
        self.video_completeness_verified = 0
        self.video_completed = False
        self.key_frame_index = (
            1  # 0 is reserved for the video completeness verification
        )
        self.camera_fps = self.calibrated_fps
        self.zip_file_path = None

        # Initialize logging
        self.logger.info(
            "Initializing Submission Client with package path: {}".format(
                self.package_path
            )
        )
        self.logger.info(f"Team Unique Code: {self.team_unique_code}")
        self.logger.info(
            f"Mode: {'Teleop' if self.autonomy_level==0 else 'Human-in-the-loop' if self.autonomy_level==1 else 'Autonomous'}"
        )
        self.logger.info(f"Video will be recorded from ROS topic: {self.camera_topic}")
        self.logger.info(f"Video and Log will be saved at Directory: {self.file_dir}")
        self.logger.info(
            f"Connecting to Server IP: {self.server_ip}, Server Port: {self.server_port}"
        )

        # Initialize socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((self.server_ip, self.server_port))
            self.client_socket.settimeout(
                30
            )  # For oversea connection we need to set a longer timeout
            self.logger.info("Connected to server successfully")
            self.connection_status = True  # Set initial connection status
            self.last_connection_check_time = time.time()

        except ConnectionRefusedError as e:
            self.logger.error(
                f"Connection refused: {e}. This is likely due to the server not running."
            )
            exit()

        except TimeoutError as e:
            self.logger.error(
                f"Connection timed out: {e}. Check network/firewall condition."
            )
            exit()

        except socket.gaierror as e:
            self.logger.error(f"Invalid hostname or DNS lookup failed: {e}")
            exit()

        except socket.error as e:
            self.logger.error(
                f"General socket error: {e}, please check your network condition"
            )
            exit()

        # Initialize execution status services
        self.task_finished_service = rospy.Service(
            ROS_TOPIC_TASK_FINISHED, Trigger, self.handle_task_finished
        )
        self.task_skipped_service = rospy.Service(
            ROS_TOPIC_TASK_SKIPPED, Trigger, self.handle_task_skipped
        )

        # Initialize connection status publisher
        self.connection_status_pub = rospy.Publisher(
            ROS_TOPIC_CONNECTION_STATUS, Bool, queue_size=10
        )
        self.task_status_pub = rospy.Publisher(
            ROS_TOPIC_ONGOING_TASK, String, queue_size=10
        )

        self.rate = rospy.Rate(100)
        self.connection_status = False

    def hard_hash_image(self, img) -> str:
        """
        Hash the image file with the one-time submission code to a 256-bit hash code
        """
        assert self.one_time_code is not None, "One-time submission code is not set"
        _, png_encoded = cv2.imencode(".png", img)
        img_bytes = png_encoded.tobytes()
        base_hash = hashlib.sha256(img_bytes).hexdigest()
        combined = (self.one_time_code + base_hash).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def hard_hash_video(self, video_path: str) -> str:
        """
        Hash the video file with the one-time submission code to a 256-bit hash code
        """
        assert self.one_time_code is not None, "One-time submission code is not set"
        hasher = hashlib.sha256()
        with open(video_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        video_hash = hasher.hexdigest()

        combined = (self.one_time_code + video_hash).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def camera_callback(self, msg):
        """
        Callback function for the camera topic
        """
        if not self.is_recording:
            self.buffer_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if cv_image is None or cv_image.size == 0:
                self.logger.error("Received empty image from camera.")
                return

            # Update the buffer frame
            self.buffer_frame = self.add_timestamp_to_image(cv_image)

            # Initialize video writer if needed
            if self.video_writer is None:
                size = (cv_image.shape[1], cv_image.shape[0])  # Get the image size
                self.video_writer = cv2.VideoWriter(
                    self.video_path,
                    cv2.VideoWriter_fourcc(*"avc1"),
                    self.camera_fps,
                    size,
                    True,
                )

                if not self.video_writer.isOpened():
                    self.logger.error("Failed to open video writer.")
                    self.video_writer = None
                    return

                # Hash code for the first frame
                response = self.register_key_frame_hash()
                if response is None:
                    self.logger.error(
                        "Failed to verify the first frame from the camera"
                    )
                    self.video_writer = None
                    return
                # Recording starts now
                if (
                    response.type == "hashcode_response"
                    and response.message == "Camera_Verified"
                ):
                    self.camera_verified = True
                    self.logger.info(
                        "Camera connection is verified for video recording."
                    )
                    self.logger.info(f"Started recording video to: {self.video_path}")
                    self.video_writer.write(cv_image)

                else:
                    self.logger.error(
                        "Failed to verify the first frame from the camera"
                    )
                    self.video_writer = None
                    return
            elif self.video_writer is not None and self.camera_verified:
                # Write frame to video
                self.video_writer.write(cv_image)
        except Exception as e:
            self.logger.error(f"Error in camera callback: {e}")

    def start_recording(self) -> bool:
        """
        Start recording the video
        """
        if self.is_recording:
            return False
        # Create video filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(
            self.file_dir, f"{self.team_unique_code}_{timestamp}.mp4"
        )
        self.video_path_compressed = os.path.join(
            self.file_dir, f"{self.team_unique_code}_{timestamp}_compressed.mp4"
        )
        self.is_recording = True
        self.start_time = datetime.now()
        # Start connection monitor thread
        self._connection_monitor_thread = threading.Thread(
            target=self.connection_monitor_thread, daemon=True
        )
        self._connection_monitor_thread.start()

        return True

    def stop_recording(self):
        """
        Stop recording the video
        """
        if not self.is_recording:
            return False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.end_time = datetime.now()
        self.logger.info("Stopped recording video.")

        return True

    def compress_video_and_log_into_zip(self, upload_path: str) -> None:
        """
        Compress the video and log file into a zip file for submission
        """
        zip_file_path = os.path.join(self.file_dir, "submission.zip")
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            zipf.write(self.log_file_path, os.path.basename(self.log_file_path))
            zipf.write(upload_path, os.path.basename(upload_path))
            for i in range(1, self.key_frame_index):
                key_frame_path = os.path.join(
                    self.file_dir, "keyframe_{}.png".format(i)
                )
                if os.path.exists(key_frame_path):
                    zipf.write(key_frame_path, os.path.basename(key_frame_path))
        self.zip_file_path = zip_file_path

    def send_request(self, request):
        """
        Send a request as JSON with a 4-byte length prefix
        """
        try:
            payload = request.model_dump_json().encode("utf-8")
            length_prefix = struct.pack(
                HEADER_FMT, len(payload)
            )  # 4-byte, network order
            self.client_socket.sendall(length_prefix + payload)
        except socket.error as e:
            self.logger.error(f"Failed to send request: {e}")
            return False
        return True

    def _recvall(self, n: int) -> bytes:
        """
        Keep reading from the socket until the specified number of bytes are received
        """
        data = bytearray()
        while len(data) < n:
            chunk = self.client_socket.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def receive_response(self, message_type=None):
        """
        Receive a length-prefixed JSON response and parse it
        """
        try:
            raw_length = self._recvall(HEADER_SIZE)
            if not raw_length:
                self.logger.error("Failed to receive message length prefix")
                return None
            msg_length = struct.unpack(HEADER_FMT, raw_length)[0]
            raw_data = self._recvall(msg_length)
            if not raw_data:
                self.logger.error("Failed to receive data stream from the server")
                return None

            data = raw_data.decode("utf-8")
            if message_type is None:
                return ServerResponse.validate_json(data)
            else:
                return message_type.model_validate_json(data)
        except socket.error as e:
            self.logger.error(f"Failed to receive response: {e}")
            return None

    def upload_video(self, video_compressed: bool) -> bool:
        """
        Upload the video to the server after receiving the presigned URL
        """
        if not self.video_completed:
            self.logger.error("Video completeness is not verified to upload")
            return False

        if video_compressed:
            if not os.path.exists(self.video_path_compressed):
                self.logger.error(
                    "Compressed video file is not found, please contact the organizers"
                )
                return False
            upload_path = self.video_path_compressed
        else:
            if not os.path.exists(self.video_path):
                self.logger.error(
                    "Original video file is not found, please contact the organizers"
                )
                return False
            upload_path = self.video_path
        try:
            # Get file size
            file_size = os.path.getsize(self.video_path)

            # Send upload command
            if file_size > 0:
                self.send_request(
                    SubmissionRequest(type="submission_request", finished=True)
                )

            # Wait for server response
            response = self.receive_response()
            if response is None:
                self.logger.error(
                    "Server not ready for upload, please contact the organizers."
                )
                return False

            if response.success == False:
                self.logger.error(
                    "Server not ready for upload, please contact the organizers. Server Response: {}".format(
                        response.message
                    )
                )
                return False
            else:
                self.logger.info("The server is ready to receive the submission file.")
            self.compress_video_and_log_into_zip(upload_path)
            self.print_box(
                "The uploading process is about to start... This may take a while."
            )
            # Parse the presigned URL
            presigned_url = response.upload_url
            self.logger.info(
                "Uploading submission file to the presigned URL: {}".format(
                    presigned_url
                )
            )

            # Upload the zip file to the presigned URL
            headers = {"Content-Type": "application/zip"}

            with open(self.zip_file_path, "rb") as f:
                upload_response = requests.put(presigned_url, data=f, headers=headers)

            if upload_response.status_code == 200:
                self.logger.info("✅ Submission Upload successful!")
            else:
                self.logger.error(
                    f"Upload failed: {upload_response.status_code}, please contact the organizers"
                )
                self.logger.error(upload_response.text)
                return False

            return True

        except Exception as e:
            self.logger.error(
                f"Error during video upload: {e}, please contact the organizers"
            )
            return False

    def run(self):
        """
        Main function to run the submission client
        """
        # Send team ID to server and check if the team is authenticated
        self.logger.info(
            "Do you accept the terms and conditions as detailed at https://manipulation-net.org/terms_and_conditions.html ? [yes/others]"
        )
        print(
            "Please type in 'yes' and press 'Enter' to accept the terms and conditions OR type anything else and press 'Enter' to reject it."
        )
        user_input = sys.stdin.readline().strip()
        if user_input.upper() != "YES":
            self.logger.info(
                "You have not accepted the terms and conditions, exiting..."
            )
            return

        with self._lock:
            try:
                self.send_request(
                    LoginRequest(
                        type="login_request",
                        team_unique_code=self.team_unique_code,
                        autonomy_level=self.autonomy_level,
                        connection_test=False,
                    )
                )
                login_response = self.receive_response()

                authenticated = True if login_response.success == 1 else False
            except Exception as e:
                self.logger.error(
                    f"Error during team ID verification: {e}, please contact the organizers with the terminal screenshot and the log file"
                )
                return

        self.logger.info("{}".format(login_response.message))
        if not authenticated:
            self.logger.error("Submission has not been approved, exiting...")
            self.connection_status = False
            self.connection_status_pub.publish(Bool(data=False))
            return
        else:
            # Parse the one-time submission code
            self.one_time_code = login_response.one_time_code
            self.print_box("Submission Registered Successfully!")
            self.print_box(
                "[ACTION REQUIRED] Write down the one-time submission code: {}".format(
                    self.one_time_code
                )
            )

        with self._lock:
            self.logger.info(f"Requesting scoring details from server...")
            self.send_request(
                TaskRequest(type="task_request", message=self.team_unique_code)
            )
            task_response = self.receive_response()
            self.scoring_details = task_response.task_details
            self.benchmark_name = task_response.benchmark_name
            self.scoring_details_list = list(self.scoring_details.keys())
            self.total_tasks_number = len(self.scoring_details_list)
            self.instruction_enabled = task_response.instruction_enabled
            self.vision_instruction_overlay = task_response.overlay_enabled
            self.assistance_allowed = task_response.assistance_allowed
            if task_response.message:
                self.logger.info("Server message: {}".format(task_response.message))

        if self.benchmark_name in AUTONOMOUS_ONLY_TASKS and (self.autonomy_level != 2):
            self.logger.error(
                f"{self.benchmark_name} task only supports the fully autonomous mode, submission will exit"
            )
            exit()

        if self.benchmark_name in ["cable_management"]:
            self.logger.info("Overwriting ROS topic for cable_management task: 'current_language_instruction' -> 'board_configuration'")
            global ROS_TOPIC_LANGUAGE_INSTRUCTION
            ROS_TOPIC_LANGUAGE_INSTRUCTION = "/mnet_client/board_configuration"

        if self.instruction_enabled:
            self.language_pub = rospy.Publisher(
                ROS_TOPIC_LANGUAGE_INSTRUCTION, String, queue_size=10
            )
            self.vision_pub = rospy.Publisher(
                ROS_TOPIC_VISION_INSTRUCTION, Image, queue_size=10
            )

            if self.vision_instruction_overlay:
                apriltag_detected = detect_apriltag(self.buffer_frame, self.cam_K)
                if not apriltag_detected:
                    self.logger.error(
                        "AprilTag is not detected. Please ensure the AprilTag is clearly visible in the camera view from the very beginning."
                    )
                    exit()
                else:
                    self.det, self.tag_id, self.corners, self.R_cw_cv, self.t_cw_cv = (
                        apriltag_detected
                    )
                    self.logger.info(f"AprilTag is detected. Tag ID: {self.tag_id}")

                self.send_request(
                    CameraConfigRequest(
                        type="camera_config_request",
                        task_config={
                            "cam_K": self.cam_K.tolist(),
                            "cam_H": self.cam_height,
                            "cam_W": self.cam_width,
                            "center": self.det.center.tolist(),
                            "tag_id": self.tag_id,
                            "corners": self.corners.tolist(),
                            "R_cw_cv": self.R_cw_cv.tolist(),
                            "t_cw_cv": self.t_cw_cv.tolist(),
                        },
                    )
                )
                camera_config_response = self.receive_response()
                if (
                    camera_config_response.type == "camera_config_response"
                    and camera_config_response.success
                ):
                    self.logger.info("Camera setup has been updated to the server")
                else:
                    self.logger.error(
                        "Failed to update the camera setup to the server, please contact the organizers"
                    )
                    exit()

            self.send_request(
                InstructionRequest(
                    type="instruction_request",
                    task_name=self.scoring_details_list[self.current_task_id],
                )
            )
            kickoff_response = self.receive_response()
            if (
                kickoff_response.type == "instruction_response"
                and kickoff_response.success
            ):
                self.parse_instruction(kickoff_response)
                self.logger.info(
                    "Kickoff task instruction has been updated by the server"
                )
            else:
                self.logger.error(
                    "Failed to receive instruction from the server, please contact the organizers"
                )
                exit()

        # Initialize human in the loop services
        if (
            self.autonomy_level == 1 or self.autonomy_level == 0
        ) and self.assistance_allowed:
            self.logger.info(
                "Human assistance services are initialized for benchmark: {}".format(
                    self.benchmark_name
                )
            )
            self.discrete_assistance_service = rospy.Service(
                ROS_TOPIC_DISCRETE_ASSISTANCE,
                Trigger,
                self.handle_discrete_assistance,
            )
            self.continuous_assistance_service = rospy.Service(
                ROS_TOPIC_CONTINUOUS_ASSISTANCE,
                Trigger,
                self.handle_continuous_assistance,
            )

        self.logger.info(f"Scoring details for each task: {self.scoring_details}")
        self.logger.info(
            f"[NOTICE] The estimated scored points during the task submission is not your final score. It will be verified by the official judges later."
        )
        self.print_box(
            "[ACTION REQUIRED] Call 'current_task_finished' or 'current_task_skipped' service to update the execution status"
        )
        if self.autonomy_level == 1 and self.assistance_allowed:
            self.print_box(
                "[ACTION REQUIRED] Call 'discrete_assistance_update' or 'continuous_assistance_update' service to update the assistance actions"
            )
        # Update connection status
        self.connection_status = True
        self.connection_status_pub.publish(Bool(data=True))

        self.logger.info(
            "Check the current on-going task at ROS topic: /mnet_client/ongoing_task"
        )
        self.logger.info(
            "Connection status is updated every 10 seconds at ROS topic: /mnet_client/connection_status"
        )

        # Start recording
        self.start_recording()
        self.logger.info("Recording started. ")

        while (not self.camera_verified) and not rospy.is_shutdown():
            self.rate.sleep()

        self.print_box(
            "After writing down: {}, please press 'Enter' to continue...".format(
                self.one_time_code
            )
        )

        while self.code_writing_awaiting and not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0.0)[0]:
                user_input = sys.stdin.readline().strip()
                self.logger.info("Continuing after writing down the code...")
                self.code_writing_awaiting = False
            self.rate.sleep()

        self.print_box(
            "After demonstrating the required objects, please press 'Enter' to continue..."
        )

        while self.detail_demonstration_awaiting and not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0.0)[0]:
                user_input = sys.stdin.readline().strip()
                self.logger.info("Preparatory work finished. Please start the task...")
                self.detail_demonstration_awaiting = False
            self.rate.sleep()

        self._task_execution_status_thread = threading.Thread(
            target=self.task_execution_status_monitor, daemon=True
        )
        self._task_execution_status_thread.start()

        if self.instruction_enabled:
            self._task_instruction_thread = threading.Thread(
                target=self.task_instruction_thread, daemon=True
            )
            self._task_instruction_thread.start()
            self.logger.info(
                "Task instruction thread started, now you can check the current task prompts at ROS topic: /mnet_client/current_language_instruction and /mnet_client/current_vision_instruction"
            )

        self.print_box("The time limit is 180 minutes.")
        self.print_box("Type 'FINISH' and press Enter to stop recording...")

        # Wait while recording
        while self.is_recording and not rospy.is_shutdown():
            # Check time limit
            if (
                datetime.now() - self.start_time
            ).total_seconds() > 180 * 60.0:  # 60 minutes
                self.logger.info("180 minutes reached, stopping recording...")
                self.stop_recording()
                break

            if select.select([sys.stdin], [], [], 0.0)[
                0
            ]:  # Check if there's input available
                user_input = sys.stdin.readline().strip()
                if user_input.upper() == "FINISH":
                    self.logger.info("Stopping recording by manual input 'FINISH'...")
                    self.stop_recording()
                else:
                    self.logger.error(
                        "Invalid input. Please type 'FINISH' to stop recording."
                    )

            # Check if all tasks are completed
            if self.current_task_id == self.total_tasks_number:
                self.logger.info("All tasks completed. Stopping recording...")
                self.stop_recording()
                break

            self.rate.sleep()

        # Wait a moment for the last frame to be processed
        rospy.sleep(1.0)
        if self.nvenc_enabled:
            self.logger.info("Using NVIDIA NVENC for video compression")
            try:
                video_compressed = self.compress_video()
            except Exception as e:
                self.logger.error(
                    f"Error during video compression: {e}, will upload the original video"
                )
                video_compressed = False
        else:
            video_compressed = False
        # Send video hash for completeness verification
        if video_compressed and os.path.exists(self.video_path_compressed):
            video_hash = self.hard_hash_video(self.video_path_compressed)
        else:
            video_hash = self.hard_hash_video(self.video_path)

        # Send video hash for video completeness verification
        self.send_request(
            HashUpdateRequest(
                type="hashcode_request",
                hashcode=video_hash,
                order=self.video_completeness_verified,
            )
        )
        # Receive the server's response
        response = self.receive_response()
        if (
            response.type == "hashcode_response"
            and response.message == "Video_Completeness_Verified"
        ):
            self.video_completed = True
            self.logger.info("Video Completeness Verified: Ready to upload.")
            self.print_box(
                "Ready to upload the submission file: Do you want to upload or discard it?"
            )
            print(
                "Please type anything and press 'Enter' to proceed the submission OR type 'discard' and press 'Enter' to discard it."
            )
            # Upload video to AWS S3 server
            user_input = sys.stdin.readline().strip()
            if user_input.upper() == "DISCARD":
                self.logger.info(
                    "Discarding this submission... You can still review the recorded video locally."
                )
                self.send_request(ShutdownRequest(type="shutdown_request"))
                return
            else:
                uploading_result = self.upload_video(video_compressed)
                if uploading_result:
                    self.logger.info(
                        "Performance file has been uploaded to the server successfully."
                    )
                else:
                    self.logger.error(
                        "Failed to upload the performance file. Please try to upload through the website: https://manipulation-net.org/submission_portal.html or contact the organizers"
                    )
        else:
            self.logger.error(
                "Video Completeness Verification Failed. Submission will be rejected."
            )
        # Let the server know that the submission is finished by all means
        self.send_request(ShutdownRequest(type="shutdown_request"))

    def register_key_frame_hash(self):
        key_frame_path = os.path.join(
            self.file_dir, "keyframe_{}.png".format(self.key_frame_index)
        )
        cv2.imwrite(key_frame_path, self.buffer_frame)
        key_frame = cv2.imread(key_frame_path, cv2.IMREAD_UNCHANGED)
        key_frame_hash = self.hard_hash_image(key_frame)
        # Let the server know that the camera is working now
        with self._lock:
            self.send_request(
                HashUpdateRequest(
                    type="hashcode_request",
                    hashcode=key_frame_hash,
                    order=self.key_frame_index,
                )
            )
            self.key_frame_index += 1
            response = self.receive_response()
        return response

    def handle_task_finished(self, req) -> TriggerResponse:
        """
        Handle the service request to mark current task as finished
        """
        try:
            if self.current_task_id == self.total_tasks_number:
                self.logger.info("All tasks completed. Skipping current request...")
                return TriggerResponse(
                    success=False,
                    message="All tasks completed. Skipping current request...",
                )
            # Get the current task name
            task_name = self.scoring_details_list[self.current_task_id]
            # Check if the task has been already reported
            assert (
                task_name not in self.finished_tasks
            ), "The status of Task {} has been already reported".format(task_name)
            self.logger.info("Updating Task {} as FINISHED ...".format(task_name))
            # Send the execution status to the server
            with self._lock:
                self.send_request(
                    ExecutionStatusRequest(
                        type="status_request",
                        task_name=task_name,
                        task_status="finished",
                    )
                )
                # Wait for server response
                response = self.receive_response()
            # Check if the server's response is successful
            if response is not None and response.success == True:
                self.finished_tasks.append(task_name)
                self.collected_points += self.scoring_details[task_name]
                self.logger.info(
                    "{} Total points: {}".format(
                        response.message, self.collected_points
                    )
                )
                # Increment the task ID
                self.current_task_id += 1
            else:
                self.logger.error(
                    "Task {} is NOT registered as FINISHED.".format(task_name)
                )
                return TriggerResponse(success=False, message=response.message)

            if self.instruction_enabled:
                self.update_task_instruction()

            self.register_key_frame_hash()
            # Return the server's response
            return TriggerResponse(success=True, message=response.message)
        except Exception as e:
            self.logger.error(f"Error marking task as finished: {e}")
            return TriggerResponse(success=False, message=str(e))

    def handle_task_skipped(self, req) -> TriggerResponse:
        """
        Handle the service request to mark current task as skipped
        """
        try:
            if self.current_task_id == self.total_tasks_number:
                self.logger.info("All tasks completed. Skipping current request...")
                return TriggerResponse(
                    success=False,
                    message="All tasks completed. Skipping current request...",
                )

            task_name = self.scoring_details_list[self.current_task_id]
            assert (
                task_name not in self.finished_tasks
            ), "The status of Task {} has been already reported".format(task_name)
            self.logger.info("Updating Task {} as SKIPPED ...".format(task_name))
            with self._lock:
                self.send_request(
                    ExecutionStatusRequest(
                        type="status_request",
                        task_name=task_name,
                        task_status="skipped",
                    )
                )
                # Wait for server response
                response = self.receive_response()
            # Check if the server's response is successful
            if response is not None and response.success == True:
                self.finished_tasks.append(task_name)
                # No points for skipped tasks
                self.logger.info(
                    "{} Total points: {}".format(
                        response.message, self.collected_points
                    )
                )
                self.current_task_id += 1
            else:
                self.logger.error(
                    "Task {} is NOT registered as SKIPPED.".format(task_name)
                )
                return TriggerResponse(success=False, message=response.message)

            if self.instruction_enabled:
                self.update_task_instruction()

            self.register_key_frame_hash()
            # Return the server's response
            return TriggerResponse(success=True, message=response.message)
        except Exception as e:
            self.logger.error(f"Error marking task as skipped: {e}")
            return TriggerResponse(success=False, message=str(e))

    def handle_discrete_assistance(self, req) -> TriggerResponse:
        """
        Handle the service request to update the discrete assistance
        """
        try:
            if self.current_task_id == self.total_tasks_number:
                self.logger.info("All tasks completed. Skipping current request...")
                return TriggerResponse(
                    success=False,
                    message="All tasks completed. Skipping current request...",
                )

            # Get the current task name
            task_name = self.scoring_details_list[self.current_task_id]
            # Send the assistance update to the server
            with self._lock:
                self.send_request(
                    AssistanceRequest(
                        type="assistance_request",
                        task_name=task_name,
                        assistance_type="discrete",
                        assistance_action="involved",
                    )
                )
                # Wait for server response
                response = self.receive_response()

            if response is not None and response.success == True:
                self.logger.info("{}".format(response.message))
            else:
                # Check if the server's response is successful
                self.logger.error(
                    "Discrete Assistance is NOT updated during task {}.".format(
                        task_name
                    )
                )
                return TriggerResponse(success=False, message=response.message)

            # Return the server's response
            return TriggerResponse(success=True, message=response.message)
        except Exception as e:
            self.logger.error(f"Error updating discrete assistance: {e}")
            return TriggerResponse(success=False, message=str(e))

    def handle_continuous_assistance(self, req) -> TriggerResponse:
        """
        Handle the service request to update the continuous assistance
        """
        try:
            if self.current_task_id == self.total_tasks_number:
                self.logger.info("All tasks completed. Skipping current request...")
                return Trigger.Response(
                    success=False,
                    message="All tasks completed. Skipping current request...",
                )
            # Get the current task name
            task_name = self.scoring_details_list[self.current_task_id]
            # Send the assistance update to the server
            with self._lock:
                self.send_request(
                    AssistanceRequest(
                        type="assistance_request",
                        task_name=task_name,
                        assistance_type="continuous",
                        assistance_action=(
                            "stopped"
                            if self.continuous_assistance_enabled
                            else "started"
                        ),
                    )
                )
                # Wait for server response
                response = self.receive_response()
            # Check if the server's response is successful
            if response is not None and response.success == True:
                self.logger.info("{}".format(response.message))
                if not self.continuous_assistance_enabled:
                    self.continuous_assistance_enabled = True
                else:
                    self.continuous_assistance_enabled = False
            else:
                # Check if the server's response is successful
                self.logger.error(
                    "Continuous Assistance is NOT updated during task {}.".format(
                        task_name
                    )
                )
                return TriggerResponse(success=False, message=response.message)

            # Return the server's response
            return TriggerResponse(success=True, message=response.message)
        except Exception as e:
            self.logger.error(f"Error updating discrete assistance: {e}")
            return TriggerResponse(success=False, message=str(e))

    def check_connection(self) -> bool:
        """
        Check if the socket connection is still alive
        """
        try:
            # Try to send a small ping message
            self.send_request(PingRequest(type="ping_request"))
            return True
        except (socket.error, OSError):
            return False

    def task_execution_status_monitor(self):
        """
        Thread function to monitor taskexecution status
        """
        while (
            self.current_task_id < self.total_tasks_number and not rospy.is_shutdown()
        ):
            try:
                # Publish connection status
                self.task_status_pub.publish(
                    String(
                        "Current Task: {}".format(
                            self.scoring_details_list[self.current_task_id]
                        )
                    )
                )

                # Sleep for a short interval
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(
                    f"Error in task execution status monitor monitor: {e}"
                )
                time.sleep(1.0)

    def task_instruction_thread(self):
        """
        Thread function to publish task instructions
        """
        while self.is_recording and not rospy.is_shutdown():
            with self._lock:
                current_vision_instruction = self.current_vision_instruction
                current_language_instruction = self.current_language_instruction

            self.language_pub.publish(String(data=current_language_instruction))
            if current_vision_instruction is not None:
                if not self.vision_instruction_overlay:
                    self.vision_pub.publish(
                        self.bridge.cv2_to_imgmsg(
                            current_vision_instruction, encoding="bgr8"
                        )
                    )
                else:
                    self.vision_pub.publish(
                        self.bridge.cv2_to_imgmsg(
                            self.overlay_rgba_on_bgr(
                                self.buffer_frame, current_vision_instruction
                            ),
                            encoding="bgr8",
                        )
                    )

            else:
                no_image = Image()
                no_image.header.stamp = rospy.Time.now()
                no_image.height = 0
                no_image.width = 0
                no_image.encoding = "mono8"  # required field
                no_image.is_bigendian = 0
                no_image.step = 0
                no_image.data = []  # empty array
                self.vision_pub.publish(no_image)
            time.sleep(0.5)

    def connection_monitor_thread(self):
        """
        Thread function to periodically check connection and handle key frame requests
        """
        while self.is_recording and not rospy.is_shutdown():
            current_time = time.time()
            if (
                current_time - self.last_connection_check_time
                >= self.heartbeat_interval
            ):
                self.last_connection_check_time = current_time
                response = None
                try:
                    with self._lock:
                        is_connected = self.check_connection()
                        if is_connected:
                            response = self.receive_response()
                    # If the server requests a key frame
                    if response is not None and response.keyframe_requested:
                        if self.buffer_frame is not None:
                            response = self.register_key_frame_hash()
                            if (
                                response is not None
                                and response.message == "Key_Frame_Registered"
                            ):
                                self.logger.info(
                                    "Random key frame verified successfully"
                                )
                            else:
                                self.logger.error(
                                    "Key frame verification failed: {}."
                                ).format(response.message)
                                continue
                        else:
                            self.logger.warning(
                                "Key frame request received but no buffer frame available."
                            )
                    # Update connection status if changed
                    if is_connected != self.connection_status:
                        self.connection_status = is_connected
                        if is_connected:
                            self.logger.info("Connection to server is active")
                        else:
                            self.logger.warning("Connection to server is lost")
                    # Publish connection status
                    self.connection_status_pub.publish(
                        Bool(data=self.connection_status)
                    )

                except Exception as e:
                    self.logger.error(f"Error in connection monitor: {e}")
                    self.connection_status = False
                    self.connection_status_pub.publish(Bool(False))
            time.sleep(0.1)

    def update_task_instruction(self):
        """
        Update the task instruction
        """
        if self.current_task_id == self.total_tasks_number:
            return
        with self._lock:
            self.send_request(
                InstructionRequest(
                    type="instruction_request",
                    task_name=self.scoring_details_list[self.current_task_id],
                )
            )
            instruction_response = self.receive_response()
            self.parse_instruction(instruction_response)
        self.logger.info("Task instructions has been updated by the server")

    def __del__(self):
        """
        Destructor to stop recording and close the socket connection
        """
        if self.is_recording:
            self.stop_recording()
        if self.client_socket:
            self.client_socket.close()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Client closed.")
