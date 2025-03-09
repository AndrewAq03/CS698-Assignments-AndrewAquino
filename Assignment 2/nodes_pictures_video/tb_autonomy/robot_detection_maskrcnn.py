#!/usr/bin/env python3

"""
Andrew Aquino

Autonomy node for the TurtleBot.

This node will subscribe to the camera feed of the robot then use the pre-trained model of detectron-maskrcnn
to predict what the objects are in the maze. 

"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import rclpy  # Import the ROS 2 Python client library
from rclpy.node import Node  # Import the Node class from ROS 2
import cv2  # Import OpenCV for computer vision tasks
from cv_bridge import CvBridge  # Import CvBridge to convert between ROS and OpenCV images
from cv_bridge import CvBridgeError
from sensor_msgs.msg import Image  # Import the Image message type from sensor_msgs
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')  # Initialize the Node with the name 'camera_node'
        
        # Subscribe to the robot's camera topic 
        self.subscription = self.create_subscription(
            Image, 
            'camera/image_raw',  
            self.image_callback, 
            10  
        )

        # This is a publisher of the image for the model to detect objects
        self.publisher_ = self.create_publisher(Image, 'detection_image', 10)  

        self.bridge = CvBridge()  # Initialize the CvBridge to convert between ROS and OpenCV images

        # Load Detectron2 model configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #A thresholdor detection
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") #used pretrained weights for the model
        self.predictor = DefaultPredictor(self.cfg)  # Initialize the predictor

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
            return

        # This will perform object detection on the video
        outputs = self.predictor(cv_image)

        # Visualize the results using Detectron2's Visualizer
        v = Visualizer(cv_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        instances = outputs["instances"].to("cpu")
        v = v.draw_instance_predictions(instances)
        result_frame = v.get_image()[:, :, ::-1]

        # Convert the processed OpenCV image back to a ROS Image message
        try:
            msg_result = self.bridge.cv2_to_imgmsg(result_frame, encoding="bgr8")
            self.publisher_.publish(msg_result)  
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert result image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)  
    node = CameraNode()  
    try:
        rclpy.spin(node)  
    except KeyboardInterrupt:
        pass  
    finally:
        node.destroy_node()  
        rclpy.shutdown()  

if __name__ == '__main__':
    main()  
