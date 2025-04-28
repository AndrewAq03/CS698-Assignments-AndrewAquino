import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from machinevisiontoolbox import CentralCamera
from spatialmath import SE3

class ObjectPoseEstimator(Node):
    def __init__(self):
        super().__init__('object_pose_estimator_mvt')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)

        self.camera_model = None
        self.object_points_3d = np.array([  
            [0.1, 0.1, 0.0],
            [0.1, -0.1, 0.0],
            [-0.1, -0.1, 0.0],
            [-0.1, 0.1, 0.0],
            [0.1, 0.1, 0.2],
            [0.1, -0.1, 0.2],
            [-0.1, -0.1, 0.2],
            [-0.1, 0.1, 0.2]
        ], dtype=np.float32)

    def camera_info_callback(self, msg):
        if self.camera_model is None:
            fx = msg.k[0]
            fy = msg.k[4]
            cx = msg.k[2]
            cy = msg.k[5]
            self.camera_model = CentralCamera(f=(fx, fy), c=(cx, cy), image_size=(msg.width, msg.height))
            self.get_logger().info('Camera info received and camera model created.')

    def image_callback(self, msg):
        if self.camera_model is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f'Error converting image: {e}')
                return

        
            image_points_2d = self.detect_2d_points(cv_image)

            if len(image_points_2d) >= 6:
                
                try:
                    T_est = self.camera_model.estpose(self.object_points_3d, np.array(image_points_2d, dtype=np.float32))

                    
                    if T_est is not None:
                        translation = T_est.t
                        rpy = T_est.rpy()

                        self.get_logger().info(f'Estimated Translation (x, y, z): {translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}')
                        self.get_logger().info(f'Estimated RPY (roll, pitch, yaw): {rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}')
                    else:
                        self.get_logger().warn('Pose estimation failed.')

                except Exception as e:
                    self.get_logger().error(f'Error in pose estimation: {e}')

            else:
                self.get_logger().warn('Not enough 2D points detected to estimate pose.')

    def detect_2d_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)
        points_2d = [kp.pt for kp in keypoints]
        return points_2d

def main(args=None):
    rclpy.init(args=args)
    node = ObjectPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
