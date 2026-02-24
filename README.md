<!--- # ManipulationNet: Benchmarking Real-World Robotic Manipulation at Scale -->

![](docs/images/logo.png)
![](docs/images/overview.png)

## Overview

Welcome to ManipulationNet! ManipulationNet ([manipulation-net.org](https://manipulation-net.org/)) is a infrastructure to host various real-world manipulation benchmarks by 1. delivering standardized task setups worldwide; and 2. evaluating authentic task performance without time, location, and system constraints.



Select your interested benchmark task [here](https://manipulation-net.org/index.html#tasks), and get registered [here](https://manipulation-net.org/registration.html).

## News

[**2026-02-23**] Task Grasping in Clutter and Cable Management now support Teleoperation mode.

[**2026-01-01**] The mnet-client has been updated for the language-conditioned tabletop_manipulation task (CameraInfo will be required).

[**2025-11-12**] The mnet-client has been updated for the grasping_in_clutter task (CameraInfo will be required). Please check the newest docs, **update** your environment and your client before use.

[**2025-10-24**] We provided [example tasks](https://github.com/ManipulationNet/mnet_block_arrangement_example_instructions) for the block arrangement benchmark, and updated the mnet-client. Please [update](https://mnet-client.readthedocs.io/ros_1/installation.html#update-your-client) your client before use.

[**2025-10-08**] Project released at [manipulation-net.org](https://manipulation-net.org/).

## What is mnet-client?

The mnet-client is a **middle layer** between the **robotic system** and the **mnet-server** to support distributed manipulation benchmarking on standardized task setups. The robotic system communicates with the mnet-client through ROS services and topics. In general, the mnet-client is responsible for:

1. **collect** authentic manipulation performance on standardized task setups and upload it to the server for comparable research;
2. **deliver** task instructions from the server to the robotic system in real-time, this could involve language/visual prompts, task-specific instructions, and more;
3. **report** task execution and human intervention logs from the robotic system to the server in real-time to better describe the manipulation performance.

## Documentation

Please refer to https://mnet-client.readthedocs.io/ros_1/ for more details about installation and usage.

We have ROS 1 and ROS 2 supported.

## Contact

If you have any questions, please do not hesitate to contact support@manipulation-net.org
