#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory
from pathlib import Path
import json
import pickle
import time
import threading
import os
import sys
from datetime import datetime

def get_workspace_root():
    """Get the root directory of the workspace"""
    return "/workspace/factr_ws"

class AutoDataRecord(Node):
    def __init__(self, name="auto_data_record_node"):
        super().__init__(name)
        
        # Set parameters
        self.declare_parameter('state_topics', [""])
        self.state_topics = self.get_parameter('state_topics').value
        
        self.declare_parameter('image_topics', [""])
        self.image_topics = self.get_parameter('image_topics').value
        
        # Debug: Check parameter values
        self.get_logger().info(f"State topics parameter: {self.state_topics}")
        self.get_logger().info(f"Image topics parameter: {self.image_topics}")
        
        self.declare_parameter('dataset_name', "auto_test")
        dataset_name = self.get_parameter('dataset_name').value
        
        # Create a folder for the current date
        current_date = datetime.now().strftime("%Y%m%d")
        self.output_dir = Path(f"{get_workspace_root()}/raw_data/{dataset_name}/{current_date}")
        
        self.declare_parameter('max_steps', 1000)
        self.max_steps = self.get_parameter('max_steps').value
        
        self.declare_parameter('target_hz', 10.0)
        self.target_hz = self.get_parameter('target_hz').value
        
        self.declare_parameter('auto_start_delay', 5.0)  # Delay time before automatic start
        self.auto_start_delay = self.get_parameter('auto_start_delay').value
        
        self.declare_parameter('save_format', 'both')  # 'json', 'pkl', 'both'
        self.save_format = self.get_parameter('save_format').value
        
        # Create output directory
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.get_logger().info(f"Saving to {self.output_dir}")
        self.get_logger().info(f"Date folder: {current_date}")
        self.get_logger().info(f"Max steps: {self.max_steps}")
        self.get_logger().info(f"Target frequency: {self.target_hz} Hz")
        self.get_logger().info(f"Auto start delay: {self.auto_start_delay} seconds")
        self.get_logger().info(f"Save format: {self.save_format}")
        
        # State management
        self.recording = False
        self.step_count = 0
        self.data_log = None
        self.pkl_data_log = None  # Data log for pickle
        self.last_record_time = 0.0  # Shared across all topics (for synchronization)
        self.latest_messages = {}  # Buffer for the latest messages
        self.record_interval = 1.0 / self.target_hz
        self.shutdown_called = False  # Prevent duplicate shutdown calls
        self.message_received = {}  # Whether a message has been received for each topic
        self.start_time = None  # Start time of recording
        
        # Timer to periodically record data from all topics
        self.record_timer = None
        
        # Topic settings
        self.topics_to_record = []
        for state_topic in self.state_topics:
            if state_topic:
                callback = self.create_callback(state_topic)
                # Determine if the topic is of type JointTrajectory
                if 'joint_trajectory' in state_topic:
                    self.create_subscription(JointTrajectory, state_topic, callback, 10)
                else:
                    self.create_subscription(JointState, state_topic, callback, 10)
                self.topics_to_record.append(state_topic)
                self.get_logger().info(f"Subscribed to state topic: {state_topic}")
        
        for image_topic in self.image_topics:
            if image_topic:
                callback = self.create_callback(image_topic)
                self.create_subscription(Image, image_topic, callback, 1)
                self.topics_to_record.append(image_topic)
                self.get_logger().info(f"Subscribed to image topic: {image_topic}")
        
        if not self.topics_to_record:
            self.get_logger().warn("No topics to record!")
        
        # Auto-start timer
        self.auto_start_timer = self.create_timer(
            self.auto_start_delay, 
            self.auto_start_recording
        )
        
        self.get_logger().info(f"Auto data collection will start in {self.auto_start_delay} seconds...")
    
    def create_callback(self, topic_name):
        """Create a callback function for the topic"""
        def callback(msg):
            # Record message reception
            self.message_received[topic_name] = True
            
            if self.recording and self.data_log is not None:
                # Save the latest message to the buffer (for synchronized recording)
                self.latest_messages[topic_name] = msg
            elif not self.recording:
                # Debug: First message when recording has not started
                if not hasattr(self, '_first_message_logged'):
                    self.get_logger().info(f"📡 Received message on {topic_name} but not recording yet")
                    self._first_message_logged = True
        return callback
    
    def record_all_topics(self):
        """Synchronize and record the latest data from all topics"""
        if not self.recording or self.data_log is None:
            return
            
        if self.step_count >= self.max_steps:
            self.stop_recording()
            return
        
        current_time = time.time()
        
        # Record the latest messages from all topics
        for topic_name, msg in self.latest_messages.items():
            self.record_data(topic_name, msg, current_time)
        
        self.step_count += 1
        
        if self.step_count % 10 == 0:
            self.get_logger().info(f"Recorded {self.step_count}/{self.max_steps} steps")
        
        # Notify on the first message reception
        if self.step_count == 1:
            self.get_logger().info(f"✅ First synchronized data recorded!")
    
    def record_data(self, topic_name, msg, timestamp):
        """Record data for individual topics"""
        # Data entry for JSON
        data_entry = {
            'timestamp': timestamp,
            'topic': topic_name,
            'step': self.step_count
        }
        
        if hasattr(msg, 'points') and len(msg.points) > 0:
            # JointTrajectory
            point = msg.points[0]
            data_entry['position'] = list(point.positions) if point.positions else []
            # data_entry['velocity'] = list(point.velocities) if point.velocities else []
            # data_entry['effort'] = list(point.effort) if point.effort else []
        elif hasattr(msg, 'position') and hasattr(msg, 'velocity') and hasattr(msg, 'effort'):
            # JointState
            data_entry['position'] = list(msg.position)
            data_entry['velocity'] = list(msg.velocity)
            data_entry['effort'] = list(msg.effort)
        elif hasattr(msg, 'effort'):
            # Torque
            data_entry['effort'] = list(msg.effort)
        elif hasattr(msg, 'data'):
            # Image
            data_entry['width'] = msg.width
            data_entry['height'] = msg.height
            data_entry['encoding'] = msg.encoding
            data_entry['data_size'] = len(msg.data)
        
        self.data_log.append(data_entry)
        
        # Data log for pickle (same format as data_record.py)
        if self.pkl_data_log is not None:
            # Data structure per topic
            if topic_name not in self.pkl_data_log["data"]:
                self.pkl_data_log["data"][topic_name] = []
                self.pkl_data_log["timestamps"][topic_name] = []
            
            # Process message data (equivalent to utils.process_msg in data_record.py)
            if hasattr(msg, 'points') and len(msg.points) > 0:
                # For JointTrajectory
                point = msg.points[0]
                processed_data = {
                    'position': list(point.positions) if point.positions else [],
                    'velocity': list(point.velocities) if point.velocities else [],
                    'effort': list(point.effort) if point.effort else []
                }
            elif hasattr(msg, 'position') and hasattr(msg, 'velocity') and hasattr(msg, 'effort'):
                # For JointState
                processed_data = {
                    'position': list(msg.position),
                    'velocity': list(msg.velocity),
                    'effort': list(msg.effort)
                }
            elif hasattr(msg, 'effort'):
                # For Torque
                processed_data = {
                    'effort': list(msg.effort)
                }
            elif hasattr(msg, 'data'):
                # For Image
                processed_data = {
                    'width': msg.width,
                    'height': msg.height,
                    'encoding': msg.encoding,
                    'data': msg.data  # Save image data as well
                }
            else:
                processed_data = str(msg)  # Other message types
            
            # Convert timestamp to nanoseconds
            time_ns = int(timestamp * 1e9)
            
            self.pkl_data_log["data"][topic_name].append(processed_data)
            self.pkl_data_log["timestamps"][topic_name].append(time_ns)
            self.pkl_data_log["all_timestamps"].append(time_ns)
    
    def auto_start_recording(self):
        """Automatically start data collection"""
        self.auto_start_timer.cancel()  # Stop the timer
        
        self.get_logger().info("🔄 Auto start timer triggered!")
        
        if not self.topics_to_record:
            self.get_logger().error("No topics to record!")
            return
        
        self.get_logger().info(f"📊 Found {len(self.topics_to_record)} topics to record")
        self.start_recording()
    
    def start_recording(self):
        """Start data collection"""
        if self.recording:
            self.get_logger().warn("Already recording!")
            return
        
        self.recording = True
        self.step_count = 0
        self.data_log = []
        self.latest_messages = {}  # Reset message buffer
        self.last_record_time = time.time()
        self.start_time = time.time()
        
        # Start the timer to periodically record all topics
        self.record_timer = self.create_timer(self.record_interval, self.record_all_topics)
        
        # Get the next index from existing files
        next_index = self.get_next_file_index()
        
        # Set file names based on the save format
        if self.save_format in ['json', 'both']:
            self.log_file = self.output_dir / f"data_log_{next_index}.json"
        if self.save_format in ['pkl', 'both']:
            self.pkl_file = self.output_dir / f"data_log_{next_index}.pkl"
        
        # Initialize data log for pickle
        if self.save_format in ['pkl', 'both']:
            self.pkl_data_log = {
                "data": {},
                "timestamps": {},
                "all_timestamps": [],
            }
            for topic in self.topics_to_record:
                self.pkl_data_log["data"][topic] = []
                self.pkl_data_log["timestamps"][topic] = []
        
        self.get_logger().info(f"🚀 Started recording")
        if self.save_format in ['json', 'both']:
            self.get_logger().info(f"JSON file: {self.log_file}")
        if self.save_format in ['pkl', 'both']:
            self.get_logger().info(f"Pickle file: {self.pkl_file}")
        self.get_logger().info(f"Target: {self.max_steps} steps at {self.target_hz} Hz")
        
        # Start a timer to check message reception status
        self.check_timeout_timer = self.create_timer(5.0, self.check_message_timeout)
    
    def get_next_file_index(self):
        """Get the next index from existing files"""
        if not self.output_dir.exists():
            return 0
        
        # Search for existing data_log_*.json and data_log_*.pkl files
        existing_files = []
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith('data_log_'):
                # Check for files in the format data_log_number.json or data_log_number.pkl
                name_parts = file_path.stem.split('_')  # ['data', 'log', 'number']
                if len(name_parts) == 3 and name_parts[0] == 'data' and name_parts[1] == 'log':
                    try:
                        index = int(name_parts[2])
                        existing_files.append(index)
                    except ValueError:
                        continue
        
        # Return the next index
        if existing_files:
            return max(existing_files) + 1
        else:
            return 0
    
    def check_message_timeout(self):
        """Check for message reception timeout"""
        if not self.recording:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Check for topics that have not received messages after 10 seconds
        if elapsed_time > 10.0:
            no_message_topics = []
            for topic in self.topics_to_record:
                if topic not in self.message_received or not self.message_received[topic]:
                    no_message_topics.append(topic)
            
            if no_message_topics:
                self.get_logger().warn(f"⚠️  No messages received from topics: {no_message_topics}")
                self.get_logger().warn("💡 Make sure teleoperation is running to generate data!")
            
            # Stop the timer (display the warning only once)
            if hasattr(self, 'check_timeout_timer'):
                self.check_timeout_timer.cancel()
    
    def stop_recording(self):
        """Stop data collection"""
        if not self.recording:
            return
        
        self.recording = False
        
        # Stop the timer
        if self.record_timer is not None:
            self.record_timer.cancel()
            self.record_timer = None
        
        # Save in JSON format
        if self.save_format in ['json', 'both'] and self.data_log:
            with open(self.log_file, 'w') as f:
                json.dump(self.data_log, f, indent=2)
            self.get_logger().info(f"✅ Saved {len(self.data_log)} records to {self.log_file}")
        
        # Save in pickle format
        if self.save_format in ['pkl', 'both'] and self.pkl_data_log:
            with open(self.pkl_file, 'wb') as f:
                pickle.dump(self.pkl_data_log, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Calculate data statistics (same format as data_record.py)
            if self.pkl_data_log['all_timestamps']:
                duration = (self.pkl_data_log['all_timestamps'][-1] - self.pkl_data_log['all_timestamps'][0]) / 1e9
                total_messages = len(self.pkl_data_log['all_timestamps'])
                self.get_logger().info(f"✅ Saved pickle data to {self.pkl_file}")
                self.get_logger().info(f"Trajectory duration: {duration:.2f}s, total messages: {total_messages}")
            else:
                self.get_logger().info(f"✅ Saved pickle data to {self.pkl_file}")
        
        if not self.data_log and not self.pkl_data_log:
            self.get_logger().warn("No data recorded!")
        else:
            self.get_logger().info(f"Total steps recorded: {self.step_count}")
        
        # Shut down the node (prevent duplicate calls)
        if not self.shutdown_called:
            self.shutdown_called = True
            self.get_logger().info("🎯 Data collection completed. Shutting down...")
            rclpy.shutdown()


def main(args=None):
    # Disable signal handling
    os.environ['RCUTILS_DISABLE_SIGNAL_HANDLERS'] = '1'
    
    rclpy.init(args=args)
    
    # Adjust log level
    import logging
    logging.getLogger('rclpy').setLevel(logging.WARNING)
    
    auto_record_node = AutoDataRecord()
    
    try:
        rclpy.spin(auto_record_node)
    except KeyboardInterrupt:
        auto_record_node.get_logger().info("Shutting down...")
    except Exception as e:
        auto_record_node.get_logger().info(f"Exception occurred: {e}")
    finally:
        auto_record_node.destroy_node()
        # Execute shutdown only if it hasn't been called already
        if not auto_record_node.shutdown_called:
            try:
                rclpy.shutdown()
            except Exception as shutdown_error:
                auto_record_node.get_logger().warn(f"Shutdown error (ignoring): {shutdown_error}")


if __name__ == '__main__':
    main()