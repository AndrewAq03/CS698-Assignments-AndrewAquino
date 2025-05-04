from typing import List


def get_help(examples: List[str]) -> str:
    """Generate a help message for the agent."""
    return f"""
        ```
        GrizzBot - Bear-y Good Robot Assistant
        ========================================

        Usage: You can interact with GrizzBot using natural language.

        Available Tools:
        - Move to Bench : Moves the robot to a predefined bench location (bench1, bench2, or bench3)
        - Get Current Location : Gets the current position of the robot
        - Stops the Robot: Stops the robot's movement
        - Describe Your Location: Tells you what the robot sees in its environment
        - execute_ros2_command: Executes a custom ROS2 command (advanced users only)

        System Commands:
        - help: Display this help message
        - examples: Show example commands
        - clear: Clear the chat history
        - exit: Exit the application
        
        ```
        """