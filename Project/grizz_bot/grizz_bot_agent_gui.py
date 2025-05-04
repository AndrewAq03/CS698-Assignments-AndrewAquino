import asyncio
import os
import atexit
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from rosa import ROSA

import rclpy
import grizz_bot_tools
from grizz_bot_prompts import get_prompts
from help import get_help

load_dotenv()


def init_ros_node_if_needed():
    if not rclpy.ok():
        rclpy.init()
    if grizz_bot_tools.node is None:
        grizz_bot_tools.node = rclpy.create_node("grizz_streamlit_agent")
        st.session_state.ros_initialized = True

@atexit.register
def shutdown_ros():
    if grizz_bot_tools.node is not None:
        grizz_bot_tools.node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


@st.cache_resource
def initialize_agent():
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        num_ctx=8192,
    )

    rosa = ROSA(
        ros_version=2,
        llm=llm,
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
        prompts=get_prompts(),
        verbose=False,
        streaming=True,
        accumulate_chat_history=True
    )
    return rosa

# --- PAGE CONFIG --- #
st.set_page_config(
    page_title="GrizzBot Assistant",
    page_icon="🐻",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

if "rosa" not in st.session_state:
    st.session_state.rosa = initialize_agent()

if "ros_initialized" not in st.session_state:
    st.session_state.ros_initialized = False


col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.title("🤖 GrizzBot 🐻")
    st.caption("Ask for help, examples, or control your robot with natural language.")

    # Show chat history
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(message)

    
    if not st.session_state.chat_history:
        with st.chat_message("bot"):
            st.markdown("Hi! I'm 🐻 GrizzBot 🤖. How can I help you today?\n\nTry typing 'help', 'examples', or ask me something about robot navigation.")
            st.session_state.chat_history.append(("bot", "Hi! I'm 🐻 GrizzBot 🤖. How can I help you today?\n\nTry typing 'help', 'examples', or ask me something about robot navigation."))

 
    if not st.session_state.chat_ended:
        user_input = st.chat_input("Ask GrizzBot something...")
    else:
        st.info("👋 GrizzBot has ended the session. Refresh the page to start again.")
        user_input = None

def handle_help():
    examples = [
        "Move to a Bench",
        "What's in front of you?",
        "What's your current position?",
        "Stop Moving",
        "Tell me a robot bear pun",
    ]
    return get_help(examples)

def handle_examples():
    examples = "\n".join([
        "1. **Move to bench_1**: Navigates to the specified bench location",
        "2. **What objects do you see?**: Reports detected objects in the robot's view",
        "3. **What's your current position?**: Reports the robot's current coordinates",
        "4. **Stop movement immediately**: Halts all robot movement",
        "5. **Move back to home position**: Returns robot to its home position",
        "6. **List all available bench locations**: Shows all navigation targets",
    ])
    return f"Here are some examples of what you can ask me:\n\n{examples}"


async def process_rosa_stream(user_message):
    response = ""
    placeholder = st.empty()
    
    async for event in st.session_state.rosa.astream(user_message):
        if event["type"] == "token":
            response += event.get("content", "")
            with placeholder.container():
                st.markdown(response)
        elif event["type"] == "final":
            final_response = event.get("content", response)
            with placeholder.container():
                st.markdown(final_response)
            response = final_response
            break
    
    return response

if user_input:
    command = user_input.strip().lower()

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))
    
    # Process special commands or send to ROSA
    with st.chat_message("bot"):
        if command == "exit":
            farewell = "👋 Goodbye! GrizzBot is shutting down."
            st.markdown(farewell)
            st.session_state.chat_history.append(("bot", farewell))
            st.session_state.chat_ended = True
            
        elif command == "help":
            help_text = handle_help()
            st.markdown(help_text)
            st.session_state.chat_history.append(("bot", help_text))
            
        elif command == "examples":
            examples_text = handle_examples()
            st.markdown(examples_text)
            st.session_state.chat_history.append(("bot", examples_text))
            
        elif command == "clear":
            st.session_state.chat_history = []
            st.session_state.rosa.clear_chat()
            st.rerun()
            
        else:
            
            if not st.session_state.ros_initialized:
                init_ros_node_if_needed()
                
            # Process with ROSA
            try:
                rosa_response = asyncio.run(process_rosa_stream(user_input))
                st.session_state.chat_history.append(("bot", rosa_response))
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(("bot", error_msg))


with st.sidebar:
    st.title("GrizzBot Status")
    
    # ROS Status
    ros_status = "✅ Connected" if st.session_state.ros_initialized else "❌ Not Connected"
    st.markdown(f"**ROS Status:** {ros_status}")
    
    # Action Buttons
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.rosa.clear_chat()
        st.rerun()
        
    if st.button("Reinitialize ROS"):
        try:
            init_ros_node_if_needed()
            st.success("ROS node reinitialized successfully")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to initialize ROS: {str(e)}")
    
    # Help and Documentation
    with st.expander("Quick Commands"):
        st.markdown("""
        - **help**: Display help information
        - **examples**: Show example commands
        - **clear**: Clear chat history
        - **exit**: End the session
        """)