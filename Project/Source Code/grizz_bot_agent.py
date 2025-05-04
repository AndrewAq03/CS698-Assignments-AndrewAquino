import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import pyinputplus as pyip
import rclpy
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain_ollama import ChatOllama
from rosa import ROSA

from grizz_bot_prompts import get_prompts
import grizz_bot_tools  
from help import get_help


load_dotenv()

class GrizzAgent:
    """
    GrizzAgent is a specialized implementation of ROSA for controlling the GrizzBot.
    
    This agent provides command-line interaction capabilities and custom commands
    for controlling the robot through natural language. 
    """
    
    def __init__(self, streaming: bool = True, verbose: bool = False):
        #Initialize LLaMA model
        self.__llm = ChatOllama(
            model="llama3.2",
            temperature=0,
            num_ctx=8192,  # Use large context size for complex queries
        )

        # self.__llm = ChatOpenAI(
        #     model_name="gpt-4o",  # or your preferred model
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),  # Using environment variable
        # )

        self.__streaming = streaming
        self.__prompts = get_prompts()
        
        # Initialize ROSA with our tools and prompts
        # self.rosa = ROSA(
        #     ros_version=2,
        #     llm=self.__llm,
        #     tools=[grizz_bot_tools],
        #     tool_packages=[],  # No additional tool packages
        #     prompts=self.__prompts,  
        #     verbose=verbose,
        #     streaming=streaming,
        #     accumulate_chat_history=True
        # )

        self.rosa = ROSA(
            ros_version=2,
            llm=self.__llm,
            tools=[
                grizz_bot_tools.move_to_bench,
                grizz_bot_tools.get_position,
                grizz_bot_tools.stop_grizz_bot,
                grizz_bot_tools.list_benches,
                grizz_bot_tools.execute_ros2_command,
                grizz_bot_tools.move_home,
                grizz_bot_tools.get_detected_objects,
            ],
            tool_packages=[],
            prompts=self.__prompts,
            verbose=verbose,
            streaming=streaming,
            accumulate_chat_history=True
        )

    #     #debugging stuff#
    #     if hasattr(self.rosa, 'get_tools') and callable(self.rosa.get_tools):
    #         print("Registered Tools:")
    #         for tool_name, tool_function in self.rosa.get_tools().items():  # Or .items() if it returns a dict
    #             print(f"  - {tool_name}: {tool_function}")
    #     else:
    #         print("ROSA object does not provide direct access to tools.")
    #     print("-" * 20)
    #  # --- End debugging---
        
        # Define example commands for help - Updated with new capabilities
        self.examples = [
            "Move to a Bench",
            "Whats in front of you?",
            "What's your current position?",
            "Stop Moving",
            "Tell me a robot bear pun",
        ]
        
        # Initialize commands dictionary
        self.commands = {
            "help": self.handle_help,
            "examples": self.handle_examples,
            "clear": self.handle_clear,
            "goodbye": self.handle_exit
        }
        
        # Set up console for rich output
        self.console = Console()
        self.last_events = []  
        
    def handle_help(self):
        """Display help information."""
        help_text = get_help(self.examples)
        self.console.print(Panel(help_text, title="Help", border_style="blue"))
        return False  
    
    def handle_examples(self):
        """Display usage examples."""
        examples_text = "Here are some example commands:\n\n"
        for i, example in enumerate(self.examples, 1):
            examples_text += f"{i}. {example}\n"
        self.console.print(Panel(examples_text, title="Examples", border_style="blue"))
        return False  
        
    def handle_clear(self):
        """Clear the chat history and console."""
        self.rosa.clear_chat()
        self.last_events = []
        os.system("clear")
        return False  
        
    def handle_exit(self):
        """Exit the application."""
        self.console.print("[bold green]Goodbye![/bold green]")
        return True  
        
    def get_input(self, prompt: str = "> "):
        """Get user input from the console."""
        return pyip.inputStr(prompt)
        
    @property
    def greeting(self):
        """Return a formatted greeting message."""
        greeting = Text("\nHi! I'm 🐻 GrizzBot 🤖. How can I help you today?\n")
        greeting.stylize("frame bold blue")
        greeting.append("Try 'help', 'examples', 'clear', or 'goodbye' ", style="italic")
        return greeting

    async def process_stream_events(self, query):
        """Process streaming events from ROSA."""
        collected_response = ""
        tool_in_progress = False
        
        
        with self.console.status("[bold green]Thinking...[/bold green]"):
            async for event in self.rosa.astream(query):
                event_type = event.get("type")
                
            #     This is for debugging purposes     
            #     if event_type == "tool_start":
            #         # --- Add these lines for debugging ---
            #         print("\n[bold magenta]Tool Start Event Data:[/bold magenta]")
            #         print(event)  # Print the entire event dictionary
            #         tool_name = event.get("name", "unknown tool")
            #         tool_input = event.get("input", {})
            #         print(f"[bold yellow]Using tool: {tool_name}[/bold yellow]")
            #         print(f"[dim]Input: {tool_input}[/dim]")
            #         tool_in_progress = True
            #  # --- End debugging lines ---

                if event_type == "token":
                    content = event.get("content", "")
                    if content:
                        if not tool_in_progress:
                            self.console.print(content, end="")
                        collected_response += content
                        
                elif event_type == "tool_start":
                    tool_name = event.get("name", "unknown tool")
                    tool_input = event.get("input", {})
                    #self.console.print(f"\n[bold yellow]Using tool: {tool_name}[/bold yellow]")
                    #self.console.print(f"[dim]Input: {tool_input}[/dim]")
                    tool_in_progress = True
                    
                elif event_type == "tool_end":
                    tool_name = event.get("name", "unknown tool")
                    tool_output = event.get("output", "No output")
                    #self.console.print(f"[bold green]Tool {tool_name} finished[/bold green]")
                    #self.console.print(f"[dim]Result: {tool_output}[/dim]\n")
                    tool_in_progress = False
                    
                elif event_type == "final":
                    content = event.get("content", "")
                    if content and content != collected_response:
                        collected_response = content
                        
                elif event_type == "error":
                    error_msg = event.get("content", "Unknown error occurred")
                    self.console.print(f"[bold red]Error: {error_msg}[/bold red]")
                    collected_response = error_msg
                    
        # Print final response in a panel
        if collected_response:
            self.console.print(Panel(collected_response, title="Response", border_style="green"))
            
        return collected_response

    async def run(self):
        """Run the agent's main interaction loop."""
        # Clear console and show greeting
        os.system("clear")
        self.console.print(self.greeting)
        
        # Initialize ROS node if required by any tools
        rclpy.init()
        node = rclpy.create_node("grizz_agent")
        
        # Store node reference for tools to use if needed
        grizz_bot_tools.node = node
        
        try:
            # Main interaction loop
            while True:
                # Get user input
                user_input = self.get_input()
                
                # Check if input is a command
                if user_input.lower() in self.commands:
                    # Execute command and check if we should exit
                    if self.commands[user_input.lower()]():
                        break
                else:
                    # Process user query
                    if self.__streaming:
                        await self.process_stream_events(user_input)
                    else:
                        # when the agent is thinking 
                        with self.console.status("[bold green]Thinking...[/bold green]"):
                            response = self.rosa.invoke(user_input)
                        #self.console.print(Panel(response, title="Response", border_style="green"))
        
        finally:
            if node:
                node.destroy_node()
            rclpy.shutdown()


def main():
    """Entry point for the application."""
    agent = GrizzAgent(streaming=True)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()