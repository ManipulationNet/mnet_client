# Implementation of the local test client for  ManipulationNet: manipulation-net.org
# DO NOT MODIFY THIS FILE
# Contact: support@manipulation-net.org

import os
import sys
import time
import json
import select
import random
import threading
from datetime import datetime

try:
    import cv2
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Bool
    from std_srvs.srv import Trigger, TriggerResponse
    from mnet_client.base import (
        BaseClient,
        AVAILABLE_TASKS,
        INSTRUCTION_ENABLED_TASKS,
        OVERLAY_ENABLED_TASKS,
        AUTONOMOUS_ONLY_TASKS,
        APRILTAG_ENABLED_TASKS,
    )
    from mnet_client.tasks import detect_apriltag, MnetSceneReplica

except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are installed and properly configured.")
    exit()


class LocalTestClient(BaseClient):
    """
    Local test client for the ManipulationNet
    """

    def __init__(self):
        super().__init__("local_test_client")
        self.print_box(
            "This is a local test client. You can experience the same task protocol locally."
        )

        # Initialize scoring details
        task_name = input(
            f"Choose the test task among {AVAILABLE_TASKS} for evaluation: "
        )  # For local testing, we need the user to input the task name
        if task_name not in AVAILABLE_TASKS:
            self.logger.error(f"Invalid task name: {task_name}")
            exit()

        self.task_name = task_name
        self.task_metadata = None
        self.scoring_details = None
        self.scoring_details_list = None
        self.language_instructions = []
        self.vision_instructions = []

        if self.task_name in INSTRUCTION_ENABLED_TASKS:
            self.instruction_enabled = True
        else:
            self.instruction_enabled = False

        if self.task_name in OVERLAY_ENABLED_TASKS:
            self.overlay_enabled = True
        else:
            self.overlay_enabled = False

        if self.task_name in APRILTAG_ENABLED_TASKS:
            if self.cam_K is None or self.cam_K.shape != (3, 3):
                self.logger.error(
                    "Camera info is not properly loaded. Please check your camera setup."
                )
                exit()
            self.logger.info("AprilTag is required for the task.")

            apriltag_detected = detect_apriltag(self.buffer_frame, self.cam_K)
            if not apriltag_detected:
                self.logger.error(
                    "AprilTag is not detected. Please ensure the AprilTag is visible in the camera image from the very beginning."
                )
                exit()
            else:
                self.det, self.tag_id, self.corners, self.R_cw_cv, self.t_cw_cv = (
                    apriltag_detected
                )
                self.logger.info(f"AprilTag is detected. Tag ID: {self.tag_id}")

        self.get_task_details()

        self.finished_tasks = []
        self.collected_points = 0
        self.current_task_id = 0

        self.total_tasks_number = len(self.scoring_details_list)
        self.continuous_assistance_enabled = False

        # 0 is reserved for the video completeness verification
        self.key_frame_index = 1
        self.camera_verified = True  # for lcoal test

        self.logger.info(
            "Initializing Submission Client with package path: {}".format(
                self.package_path
            )
        )
        self.logger.info(f"Team Unique Code: {self.team_unique_code}")
        self.logger.info(
            f"Mode: {'Teleop' if self.autonomy_level==0 else 'Human-in-the-loop' if self.autonomy_level==1 else 'Autonomous'}"
        )

        if self.task_name in AUTONOMOUS_ONLY_TASKS and (
            self.autonomy_level == 1 or self.autonomy_level == 0
        ):
            self.logger.info(
                "Human-in-the-loop and Teleoperation mode are not supported for block arrangement task"
            )
            exit()

        self.logger.info(f"Video will be recorded from ROS topic: {self.camera_topic}")
        self.logger.info(f"Video and Log will be saved at Directory: {self.file_dir}")
        self.logger.info(f"Scoring details for each task: {self.scoring_details}")

        self.logger.info(
            f"[NOTICE] The estimated score points from the local test are only for your reference. The actual scoring details will be provided by the server and available at manipulation-net.org."
        )

        # Initialize execution status services
        self.task_finished_service = rospy.Service(
            "mnet_client/current_task_finished", Trigger, self.handle_task_finished
        )
        self.task_skipped_service = rospy.Service(
            "mnet_client/current_task_skipped", Trigger, self.handle_task_skipped
        )
        self.camera_fps = self.calibrated_fps

        # Initialize human in the loop services
        if self.autonomy_level == 1 or self.autonomy_level == 0:
            self.discrete_assistance_service = rospy.Service(
                "mnet_client/discrete_assistance_update",
                Trigger,
                self.handle_discrete_assistance,
            )
            self.continuous_assistance_service = rospy.Service(
                "mnet_client/continuous_assistance_update",
                Trigger,
                self.handle_continuous_assistance,
            )

        # Initialize connection status publisher
        self.connection_status_pub = rospy.Publisher(
            "mnet_client/connection_status", Bool, queue_size=10
        )
        self.task_status_pub = rospy.Publisher(
            "mnet_client/ongoing_task", String, queue_size=10
        )
        self.connection_status = False
        self.rate = rospy.Rate(100)
        self.last_connection_check_time = time.time()

    def get_task_details(self) -> None:
        """
        Get the task details based on the user input
        """
        scoring_details_file_path = os.path.join(
            self.package_path,
            "src",
            "mnet_client",
            "assets",
            f"{self.task_name}",
            "scoring_details.json",
        )
        if os.path.exists(scoring_details_file_path):
            with open(scoring_details_file_path, "r") as f:
                self.scoring_details = json.load(f)
        else:
            self.logger.error(
                f"Scoring details file not found: {scoring_details_file_path}"
            )
            exit()

        self.scoring_details_list = list(self.scoring_details.keys())

        task_metadata_file_path = os.path.join(
            self.package_path,
            "src",
            "mnet_client",
            "assets",
            f"{self.task_name}",
            "metadata.json",
        )

        if self.task_name in ["block_arrangement"]:
            self.instruction_enabled = True
            if os.path.exists(task_metadata_file_path):
                with open(task_metadata_file_path, "r") as f:
                    self.task_metadata = json.load(f)
            else:
                self.logger.error(
                    f"Task metadata file not found: {task_metadata_file_path}"
                )
                exit()

            entry_level_tasks = {
                k: v for k, v in self.task_metadata.items() if v.get("level") == "entry"
            }
            easy_level_tasks = {
                k: v for k, v in self.task_metadata.items() if v.get("level") == "easy"
            }
            medium_level_tasks = {
                k: v
                for k, v in self.task_metadata.items()
                if v.get("level") == "medium"
            }
            hard_level_tasks = {
                k: v for k, v in self.task_metadata.items() if v.get("level") == "hard"
            }

            def load_random_task(task_id: str, task_pool: dict) -> dict:
                if task_id.startswith("L"):
                    task_rnd = random.choice(
                        list(
                            {
                                k: v
                                for k, v in task_pool.items()
                                if v.get("mode") == "L"
                            }.keys()
                        )
                    )
                elif task_id.startswith("VL"):
                    task_rnd = random.choice(
                        list(
                            {
                                k: v
                                for k, v in task_pool.items()
                                if v.get("mode") == "VL"
                            }.keys()
                        )
                    )
                else:
                    task_rnd = random.choice(
                        list(
                            {
                                k: v
                                for k, v in task_pool.items()
                                if v.get("mode") == "V"
                            }.keys()
                        )
                    )
                task = task_pool.pop(task_rnd)
                return task

            for task_id in self.scoring_details_list:
                if self.scoring_details[task_id] == 1:
                    task = load_random_task(task_id, entry_level_tasks)
                elif self.scoring_details[task_id] == 2:
                    task = load_random_task(task_id, easy_level_tasks)
                elif self.scoring_details[task_id] == 5:
                    task = load_random_task(task_id, medium_level_tasks)
                elif self.scoring_details[task_id] == 10:
                    task = load_random_task(task_id, hard_level_tasks)

                language_instruction = task["description"]
                self.language_instructions.append(
                    language_instruction if language_instruction is not None else ""
                )
                image_path = os.path.join(
                    self.package_path,
                    "src",
                    "mnet_client",
                    "assets",
                    f"{self.task_name}",
                    "images",
                    f"{task['image']}",
                )
                self.vision_instructions.append(
                    cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if image_path is not None
                    else None
                )

        elif self.task_name in ["grasping_in_clutter"]:
            self.instruction_enabled = True
            self.scene_render = MnetSceneReplica(
                self.package_path,
                self.task_name,
                self.cam_K,
                self.cam_width,
                self.cam_height,
                self.det,
                self.tag_id,
                self.corners,
                self.R_cw_cv,
                self.t_cw_cv,
            )
            if os.path.exists(task_metadata_file_path):
                with open(task_metadata_file_path, "r") as f:
                    self.task_metadata = json.load(f)
            else:
                self.logger.error(
                    f"Task metadata file not found: {task_metadata_file_path}"
                )
                exit()

            def load_random_task(task_pool: dict) -> dict:
                key = random.choice(list(task_pool.keys()))
                value = task_pool.pop(key)
                return value

            easy_level_tasks = {
                k: v for k, v in self.task_metadata.items() if v.get("level") == "easy"
            }
            medium_level_tasks = {
                k: v
                for k, v in self.task_metadata.items()
                if v.get("level") == "medium"
            }
            hard_level_tasks = {
                k: v for k, v in self.task_metadata.items() if v.get("level") == "hard"
            }

            for idx in range(len(self.scoring_details_list)):
                if idx in [0, 1, 2, 3, 4]:
                    task = load_random_task(easy_level_tasks)
                elif idx in [5, 6, 7, 8, 9]:
                    task = load_random_task(medium_level_tasks)
                else:
                    task = load_random_task(hard_level_tasks)

                self.language_instructions.append("")
                scene_id = task["layout"]
                rendered_scene_with_axis = self.render_overlay_image_based_on_scene_id(
                    scene_id
                )
                self.vision_instructions.append(rendered_scene_with_axis)

        elif self.task_name in ["tabletop_manipulation"]:
            self.instruction_enabled = True
            self.scene_render = MnetSceneReplica(
                self.package_path,
                self.task_name,
                self.cam_K,
                self.cam_width,
                self.cam_height,
                self.det,
                self.tag_id,
                self.corners,
                self.R_cw_cv,
                self.t_cw_cv,
            )
            if os.path.exists(task_metadata_file_path):
                with open(task_metadata_file_path, "r") as f:
                    self.task_metadata = json.load(f)
            else:
                self.get_logger().error(
                    f"Task metadata file not found: {task_metadata_file_path}"
                )
                exit()

            def get_layouts_by_level(tasks_dict, level):
                layouts = set()
                for task_key, task_info in tasks_dict.items():
                    if task_info.get("level") == level:
                        layouts.add(task_info.get("layout"))

                return sorted(list(layouts))

            easy_layouts = get_layouts_by_level(self.task_metadata, "easy")
            medium_layouts = get_layouts_by_level(self.task_metadata, "medium")
            hard_layouts = get_layouts_by_level(self.task_metadata, "hard")

            # unordered skill distribution for each level
            easy_level_skills = [
                "pick up",
                "pick up",
                "pick up",
                "pick up",
                "get",
                "get",
                "get",
                "push",
                "push",
                "push",
            ]
            medium_level_skills = [
                "pick up",
                "pick up",
                "pick up",
                "get",
                "push",
                "push",
                "knock over",
                "move away",
                "remove",
                "remove",
            ]
            hard_level_skills = [
                "next to",
                "next to",
                "next to",
                "next to",
                "into",
                "into",
                "stack",
                "stack",
                "stack",
                "upright",
            ]

            def get_instruction_by_layout_level_and_skill(
                tasks_dict, layout=None, level=None, skill=None, with_layout=False
            ):
                if skill is None and level is None:
                    matching_keys = [
                        key
                        for key, task_info in tasks_dict.items()
                        if task_info.get("layout") == layout
                    ]
                elif skill is None and level is not None:
                    matching_keys = [
                        key
                        for key, task_info in tasks_dict.items()
                        if task_info.get("layout") == layout
                        and task_info.get("level") == level
                    ]
                elif skill is not None and level is None:
                    matching_keys = [
                        key
                        for key, task_info in tasks_dict.items()
                        if task_info.get("layout") == layout
                        and task_info.get("skill") == skill
                    ]
                elif layout is None and level is not None and skill is not None:
                    matching_keys = [
                        key
                        for key, task_info in tasks_dict.items()
                        if task_info.get("level") == level
                        and task_info.get("skill") == skill
                    ]
                else:
                    matching_keys = [
                        key
                        for key, task_info in tasks_dict.items()
                        if task_info.get("layout") == layout
                        and task_info.get("level") == level
                        and task_info.get("skill") == skill
                    ]
                if not matching_keys:
                    return None

                selected_key = random.choice(matching_keys)
                selected_instruction = tasks_dict[selected_key].get("instruction")
                if layout is None:
                    layout = tasks_dict[selected_key].get("layout")
                del tasks_dict[selected_key]

                if with_layout:
                    return selected_instruction, layout
                else:
                    return selected_instruction

            for idx in range(len(self.scoring_details_list)):
                if (idx // 5) in [0, 1] and (idx % 5) == 0:
                    scene_id = easy_layouts.pop()
                    rendered_scene_with_axis = (
                        self.render_overlay_image_based_on_scene_id(scene_id)
                    )
                    for _ in range(5):
                        self.language_instructions.append(
                            get_instruction_by_layout_level_and_skill(
                                self.task_metadata,
                                scene_id,
                                skill=easy_level_skills.pop(
                                    random.randint(0, len(easy_level_skills) - 1)
                                ),
                            )
                        )
                        self.vision_instructions.append(rendered_scene_with_axis)

                if (idx // 5) in [2, 3] and (idx % 5) == 0:
                    scene_id = medium_layouts.pop()
                    rendered_scene_with_axis = (
                        self.render_overlay_image_based_on_scene_id(scene_id)
                    )
                    for _ in range(5):
                        self.language_instructions.append(
                            get_instruction_by_layout_level_and_skill(
                                self.task_metadata,
                                scene_id,
                                skill=medium_level_skills.pop(
                                    random.randint(0, len(medium_level_skills) - 1)
                                ),
                            )
                        )
                        self.vision_instructions.append(rendered_scene_with_axis)

                elif idx in range(20, 30):
                    skill_type = hard_level_skills.pop(
                        random.randint(0, len(hard_level_skills) - 1)
                    )
                    language_instruction, scene_id = (
                        get_instruction_by_layout_level_and_skill(
                            self.task_metadata,
                            level="hard",
                            skill=skill_type,
                            with_layout=True,
                        )
                    )
                    rendered_scene_with_axis = (
                        self.render_overlay_image_based_on_scene_id(scene_id)
                    )
                    self.language_instructions.append(language_instruction)
                    self.vision_instructions.append(rendered_scene_with_axis)

    def camera_callback(self, msg: Image) -> None:
        """
        Callback function to handle the camera topic
        """
        if not self.is_recording:
            self.buffer_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if cv_image is None or cv_image.size == 0:
                self.logger.error("Received empty image from camera.")
                return
            # Update the buffer frame
            self.buffer_frame = self.add_timestamp_to_image(cv_image)

            # Initialize video writer if not already opened
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

            elif self.video_writer is not None and self.camera_verified:
                # Write frame to video
                self.video_writer.write(cv_image)

        except Exception as e:
            self.logger.error(f"Error in camera callback: {e}")

    def start_recording(self):
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
        if self.instruction_enabled == True:
            self._task_instruction_thread = threading.Thread(
                target=self.task_instruction_thread, daemon=True
            )
            self._task_instruction_thread.start()
        return True

    def stop_recording(self) -> bool:
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

    def run(self):
        """
        Main function to run the submission client
        """

        # Start connection monitoring thread
        self._task_execution_status_thread = threading.Thread(
            target=self.task_execution_monitor_thread, daemon=True
        )
        self._task_execution_status_thread.start()
        self.logger.info(
            "Task Execution Status monitoring thread started. Check the current on-going task at ROS topic: /mnet_client/ongoing_task"
        )

        if self.instruction_enabled:
            self.language_pub = rospy.Publisher(
                "/mnet_client/current_language_instruction", String, queue_size=1
            )
            self.vision_pub = rospy.Publisher(
                "/mnet_client/current_vision_instruction", Image, queue_size=1
            )

        # Start recording
        self.start_recording()
        self.logger.info("Recording started.")

        # Start recording
        self.start_recording()
        self.logger.info("Recording started.")

        while (not self.camera_verified) and not rospy.is_shutdown():
            self.rate.sleep()

        self.print_box(
            "After writing down: {}, please press 'Enter' to continue...".format(
                "TEST1234"
            )
        )

        while self.code_writing_awaiting and not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0.0)[0]:
                user_input = sys.stdin.readline().strip()
                self.logger.info("Continuing after writing down the code...")
                self.code_writing_awaiting = False
            self.rate.sleep()

        self.print_box(
            "After demonstrating the test hardware, please press 'Enter' to continue..."
        )

        while self.detail_demonstration_awaiting and not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0.0)[0]:
                user_input = sys.stdin.readline().strip()
                self.logger.info("Preparatory work finished. Please start the task...")
                self.detail_demonstration_awaiting = False
            self.rate.sleep()

        self.print_box("The time limit is 180 minutes.")
        self.print_box("Type 'FINISH' and press Enter to stop recording...")

        # Wait while recording
        while self.is_recording and not rospy.is_shutdown():
            # Check time limit
            if (
                datetime.now() - self.start_time
            ).total_seconds() > 180 * 60.0:  # 180 minutes
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
        self.print_box(
            "Local test completed. Video is compressed: {}".format(video_compressed)
        )
        self.logger.info(
            "Local test completed. Check your video and log files in the directory: {}".format(
                self.file_dir
            )
        )

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

            task_name = self.scoring_details_list[self.current_task_id]
            assert (
                task_name not in self.finished_tasks
            ), "The completeness of Task {} has been already reported".format(task_name)
            self.logger.info("Updating Task {} as FINISHED ...".format(task_name))

            self.finished_tasks.append(task_name)
            self.collected_points += self.scoring_details[task_name]
            self.logger.info(
                "Task {} marked as FINISHED. Total points: {}".format(
                    task_name, self.collected_points
                )
            )
            self.current_task_id += 1

            # Return the server's response
            return TriggerResponse(
                success=True, message="Task {} marked as FINISHED.".format(task_name)
            )

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
            ), "The completeness of Task {} has been already reported".format(task_name)
            self.logger.info("Updating Task {} as SKIPPED ...".format(task_name))

            self.finished_tasks.append(task_name)
            # No points for skipped tasks
            self.logger.info(
                "Task {} marked as SKIPPED. Total points: {}".format(
                    task_name, self.collected_points
                )
            )
            self.current_task_id += 1

            # Return the server's response
            return TriggerResponse(
                success=True, message="Task {} marked as SKIPPED.".format(task_name)
            )

        except Exception as e:
            self.logger.error(f"Error marking task as skipped: {e}")
            return TriggerResponse(success=False, message=str(e))

    def handle_discrete_assistance(self, req) -> TriggerResponse:
        """
        Handle the service request to update the discrete assistance
        """
        try:
            # Get the current task name
            task_name = self.scoring_details_list[self.current_task_id]
            # Send the assistance update to the server

            self.logger.info(
                "Discrete Assistance involved during task {}".format(task_name)
            )
            response = "Discrete Assistance is updated during task {}.".format(
                task_name
            )
            # Return the server's response
            return TriggerResponse(success=True, message=response)
        except Exception as e:
            self.logger.error(f"Error updating discrete assistance: {e}")
            return TriggerResponse(success=False, message=str(e))

    def handle_continuous_assistance(self, req) -> TriggerResponse:
        """
        Handle the service request to update the continuous assistance
        """
        try:
            # Get the current task name
            task_name = self.scoring_details_list[self.current_task_id]

            # Check if the server's response is successful
            response = "Continuous Assistance is updated during task {}.".format(
                task_name
            )
            if not self.continuous_assistance_enabled:
                self.logger.info(
                    "Continuous Assistance started during task {}".format(task_name)
                )
                self.continuous_assistance_enabled = True
            else:
                self.logger.info(
                    "Continuous Assistance stoped during task {}".format(task_name)
                )
                self.continuous_assistance_enabled = False

            # Return the server's response
            return TriggerResponse(success=True, message=response)
        except Exception as e:
            self.logger.error(f"Error updating discrete assistance: {e}")
            return TriggerResponse(success=False, message=str(e))

    def task_instruction_thread(self):
        """
        Thread function to publish task instructions
        """
        while self.is_recording and not rospy.is_shutdown():
            with self._lock:
                current_task_id = self.current_task_id

            current_vision_instruction = self.vision_instructions[current_task_id]
            current_language_instruction = self.language_instructions[current_task_id]

            self.language_pub.publish(String(data=current_language_instruction))
            if current_vision_instruction is not None:
                if not self.overlay_enabled:
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
        Thread function to periodically check connection (not used in local test)
        """
        while self.is_recording and not rospy.is_shutdown():
            current_time = time.time()
            if (
                current_time - self.last_connection_check_time
                >= self.heartbeat_interval  # fake connection check in local test
            ):
                self.last_connection_check_time = current_time
                try:
                    # Publish connection status
                    self.connection_status_pub.publish(
                        Bool(data=self.connection_status)
                    )

                except Exception as e:
                    self.logger.error(f"Error in connection monitor: {e}")
                    self.connection_status = False
                    self.connection_status_pub.publish(Bool(data=False))
            time.sleep(0.1)

    def task_execution_monitor_thread(self):
        """
        Thread function to continuously monitor connection status
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

    def __del__(self):
        """
        Destructor to stop recording and close the socket connection
        """
        if self.is_recording:
            self.stop_recording()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Client closed.")


if __name__ == "__main__":
    """
    Main function to run the local test client
    """
    try:
        client = LocalTestClient()
        client.run()
    except rospy.ROSInterruptException:
        pass
