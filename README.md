<h2>Autonomous vision in drones: Prototyping implementation of computer vision algorithms for object detection.</h2>

This repository contains the code and resources for the final year project "Autonomous vision in drones: Prototyping implementation of computer vision algorithms for object detection." The project focuses on using the Tello drone, machine learning, and computer vision to detect objects in real-time.

Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Video Demonstration](#video-demonstration)
8. [Code Walkthrough](#code-walkthrough)
9. [License](#license)

## Introduction

In this project, we explore the capabilities of autonomous vision in drones, particularly using machine learning for object detection. The Tello drone is used as a platform for capturing real-time video, which is then processed to detect objects accurately.

## Features

- Real-time object detection using the Tello drone.
- Integration with advanced tools such as Roboflow for model training and inference.
- Comprehensive documentation and code walkthrough.

## Requirements

- Python 3.8 or higher
- Tello drone
- Wi-Fi adapter for connecting to the drone
- OpenCV
- DJITelloPy
- Inference SDK from Roboflow
- A computer that can connect to two Wi-Fi networks simultaneously
- An account on Roboflow (where you will need to input your API key)
- Selection of the detection model type, version, and API from Roboflow

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AnthonyPerez98/tfg-drone.git
    cd tfg-drone
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

1. Obtain your Roboflow API key and set up your project as described in the [Roboflow documentation](https://roboflow.com/docs).
2. Create a `.env` file in the project root directory and add your API key:
    ```env
    API_KEY=your_roboflow_api_key
    ```

## Usage

Running the main code

The primary script for object detection is `drone-sdk-detection-canV2.py`. This script performs real-time object detection using the Tello drone.

To run the script:
```sh
python drone-sdk-detection-canV2.py
```

Optional script

`drone-sdk-detection-can.py` is an optional script that can be used if you only want to retrieve the detection data without performing any further processing or visualization.

To run the optional script:
```sh
python drone-sdk-detection-can.py
```

Model selection

The code is designed to be general and can detect any objects, depending on the model chosen. You will need to select the model type, version, and API from Roboflow. This flexibility allows you to detect various objects by configuring the appropriate model in Roboflow and updating your API key and model details accordingly.

## Video Demonstration

A video demonstrating the results of the project can be found [here](https://drive.google.com/file/d/1Dujp9PP1uPRzUcGb-uTcnFvODqBWAImF/view?usp=sharing).

## Code Walkthrough

A detailed walkthrough of the code is available in the form of a video [here](https://drive.google.com/file/d/1XkySs3tZjYA0bxSWxae-JlXaWOYnZB_n/view?usp=drive_link).


---

Feel free to contribute to this project by submitting issues or pull requests. Your feedback and contributions are highly appreciated!

---

This README provides a comprehensive guide for setting up and running the project, ensuring that users can easily replicate and build upon the work done.
