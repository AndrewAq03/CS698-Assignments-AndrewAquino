#!/usr/bin/env python3

"""
Andrew Aquino

Autonomy node for the TurtleBot.

This node will subscribe to the camera feed of the robot then use the pre-trained model of detectron-maskrcnn
to predict what the objects are in the maze. This will also take the data when the object is detected 
and also the odometry data and send it to a local qdrant server.

"""


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry  # Import Odometry message
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import qdrant_client  # Import Qdrant client
from qdrant_client.http import models as qmodels  # Import Qdrant models

class ObjectDetectionQdrantNode(Node):
    def __init__(self):
        super().__init__('object_detection_qdrant_node')

        # Subscribe to camera feed
        self.image_subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # Subscribe to odometry data
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',  # Adjust topic name as needed
            self.odom_callback,
            10
        )

        self.publisher_ = self.create_publisher(Image, 'detection_image', 10)
        self.bridge = CvBridge()

        # Detectron2 setup
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

        # Qdrant setup
        self.qdrant_client = qdrant_client.QdrantClient("localhost", port=6333)  # Connect to local host of Qdrant
        self.collection_name = "object_detections_metadata"
        self.create_qdrant_collection()

        self.latest_odom = None  

    def create_qdrant_collection(self):
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={"detection_data": qmodels.VectorParams(size=128, distance=qmodels.Distance.COSINE)},
        )

    #callback functions for the odometry and image data
    def odom_callback(self, msg):
        self.latest_odom = msg

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
            return

        outputs = self.predictor(cv_image)
        instances = outputs["instances"].to("cpu")

        if len(instances) > 0:
            self.process_detections(instances)

        v = Visualizer(cv_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(instances)
        result_frame = v.get_image()[:, :, ::-1]

        try:
            msg_result = self.bridge.cv2_to_imgmsg(result_frame, encoding="bgr8")
            self.publisher_.publish(msg_result)
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert result image: {str(e)}')

    #just a error message if we don't have any data
    def process_detections(self, instances):
        if self.latest_odom is None:
            self.get_logger().warn("No odometry data available.")
            return

        #this si how we get the data from the boundry boxes the rcnn model detected
        for i in range(len(instances)):
            bbox = instances.pred_boxes[i].tensor.cpu().numpy().tolist()[0]
            label = self.cfg.MODEL.META_ARCHITECTURE.split("RCNN")[0] if instances.has("pred_classes") else "Unknown" # Get the model type
            if instances.has("pred_classes"):
              label = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[instances.pred_classes[i].item()]
            
            odom_pose = self.latest_odom.pose.pose
            odom_position = odom_pose.position
            odom_orientation = odom_pose.orientation

            # This is the payload for Qdrant
            payload = {
                "bbox": bbox,
                "label": label,
                "odom_position": {
                    "x": odom_position.x,
                    "y": odom_position.y,
                    "z": odom_position.z,
                },
                "odom_orientation": {
                    "x": odom_orientation.x,
                    "y": odom_orientation.y,
                    "z": odom_orientation.z,
                    "w": odom_orientation.w,
                },
            }

            # Create a dummy vector for Qdrant, this will be replaced with actuall data when ran
            vector = [0.0] * 128  

            # Upsert data into Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=abs(hash(str(payload))),  # Generate a unique ID
                        payload=payload,
                        vector={"detection_data": vector},
                    )
                ],
            )
            self.get_logger().info(f"Object detected and saved to Qdrant: {payload}")
            
            
def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionQdrantNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()