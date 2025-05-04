from typing import Dict
import subprocess
import yaml
import math
from pydantic import BaseModel
from langchain.tools import tool



node = None

# Load locations from YAML file
try:
    with open("locations.yaml", "r") as f:  
        locations = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: locations.yaml not found. Make sure it's in the correct directory.")
    locations = {}  

def euler_to_quaternion(theta):
    """Convert Euler angle (yaw only) to quaternion components"""
    # For 2D navigation, we only care about rotation around Z axis
    # [w, x, y, z] format
    w = math.cos(theta / 2)
    z = math.sin(theta / 2)
    return [w, 0.0, 0.0, z]

@tool("execute_ros2_command")
def execute_ros2_command(command: str) -> str:
    """
    Execute a ROS2 command through the command line.
    
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Command executed successfully: {result.stdout}"
        else:
            return f"Command failed with error: {result.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool("move_to_bench")
def move_to_bench(location_name: str) -> str:
    """
    Moves the robot to a bench using ROS2 action command line interface.
    """
    # Check if location exists
    if location_name not in locations:
        return f"Error: Location '{location_name}' not found in locations.yaml."

    # Get coordinates
    x, y, theta = locations[location_name]
    
    # Convert theta to quaternion
    quat = euler_to_quaternion(float(theta))
    
    # Build the ROS2 action command
    command = (
        f"ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \"{{pose: {{header: "
        f"{{frame_id: 'map', stamp: {{sec: 0, nanosec: 0}}}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, "
        f"orientation: {{w: {quat[0]:.4f}, x: {quat[1]:.4f}, y: {quat[2]:.4f}, z: {quat[3]:.4f}}}}}}}}}\""
    )
    
    # Execute the command
    result = execute_ros2_command(command)
    
    return f"Navigating to {location_name} (x={x}, y={y}, theta={theta})\nCommand result: {result}"

class GetPositionParams(BaseModel):
    topic: str = "/amcl_pose"

@tool("get_current_position")
def get_position(input: GetPositionParams = GetPositionParams()) -> str:
    """
    Get the current position using ROS2 command line.

    """
    # Build command to get the latest pose
    command = f"ros2 topic echo --once {input.topic}"
    result = execute_ros2_command(command)
    return f"Current position information:\n{result}"

@tool("stop_bot")
def stop_grizz_bot() -> str:
    """
    Stop the robot by publishing zero velocities to cmd_vel topic.

    """
    # Build command to publish zero velocities
    command = "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \"linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}\""
    result = execute_ros2_command(command)
    return f"Robot stopped. Command result: {result}"

@tool("list_benches")
def list_benches() -> str:
    """
    List all available bench locations.

    """
    if not locations:
        return "No locations found in locations.yaml file"
    
    result = "Available bench locations:\n"
    for name, coords in locations.items():
        result += f"- {name}: position [x={coords[0]}, y={coords[1]}], orientation [theta={coords[2]}]\n"
    
    return result

@tool("go_home")
def move_home(home_location_name: str = "Home") -> str:  
    """
    Moves the robot to a home base using ROS2 navigation.

    """
    if home_location_name not in locations:
        return f"Error: Home location '{home_location_name}' not found in locations.yaml."

    x, y, theta = locations[home_location_name]
    
    # Convert theta to quaternion
    quat = euler_to_quaternion(float(theta))
    
    # Build the ROS2 action command
    command = (
        f"ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \"{{pose: {{header: "
        f"{{frame_id: 'map', stamp: {{sec: 0, nanosec: 0}}}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, "
        f"orientation: {{w: {quat[0]:.4f}, x: {quat[1]:.4f}, y: {quat[2]:.4f}, z: {quat[3]:.4f}}}}}}}}}\""
    )
    
    # Execute the command
    result = execute_ros2_command(command)
    
    return f"Navigating Home (x={x}, y={y}, theta={theta})\nCommand result: {result}"


@tool("get_detected_objects")
def get_detected_objects(topic: str = "/detected_objects") -> str:
    """
    Get the list of currently detected objects from the object detection topic.

    """
    command = f"ros2 topic echo --once {topic}"
    result = execute_ros2_command(command)
    return f"Detected objects:\n{result}"
