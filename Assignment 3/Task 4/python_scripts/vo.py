import os
import cv2
import numpy as np
import math
from pathlib import Path
from vo_utils import getCalibrationData, featureDetection, featureTracking
from vo_h import VisualOdometry


MAX_FRAME = 1000
MIN_NUM_FEAT = 2000

def getAbsoluteScale(frame_id):
    """
    Get the absolute scale from ground truth data
    """
    dataset_path = Path.cwd().parent/ "StellaVSLAM" /"monocular_vo" / "kitti_dataset" / "data_odometry_poses/dataset/poses/00.txt"
    x, y, z = 0, 0, 0
    x_prev, y_prev, z_prev = 0, 0, 0
    i = 0
    
    if dataset_path.exists():
        with open(dataset_path, 'r') as myfile:
            for line in myfile:
                if i > frame_id:
                    break
                    
                z_prev, x_prev, y_prev = z, x, y
                values = line.strip().split()
                
                for j, value in enumerate(values):
                    z = float(value)
                    if j == 7:
                        y = z
                    if j == 3:
                        x = z
                
                i += 1
    else:
        print(dataset_path)
        print("Unable to open file")
        return 0
    
    return math.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)


def run(self, dataset_path):
    """
    Run visual odometry on dataset
    """
    # Initialize variables
    R_f = np.eye(3)  # Final rotation matrix (camera pose)
    t_f = np.zeros((3, 1))  # Final translation vector (camera pose)
    
    # Get first two images
    filename1 = os.path.join(dataset_path, f"image_0/{0:06d}.png")
    filename2 = os.path.join(dataset_path, f"image_0/{1:06d}.png")
    
    # Read the first two frames from the dataset for initial setup
    img_1_c = cv2.imread(filename1)
    img_2_c = cv2.imread(filename2)
    
    if img_1_c is None or img_2_c is None:
        print("Error reading images!")
        return -1
    
 
    img_1 = cv2.cvtColor(img_1_c, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_c, cv2.COLOR_BGR2GRAY)
    
    # Feature detection and tracking
    points1 = featureDetection(img_1)  # Detect features in img_1
    points2 = np.array([])
    points1, points2, status = featureTracking(img_1, img_2, points1, points2)
    
    # Get calibration data
    focal, pp = getCalibrationData(dataset_path)
    
    # Recover the pose and the essential matrix
    E, mask = cv2.findEssentialMat(points2, points1, focal, pp, cv2.RANSAC, 0.999, 1.0)
    _, R, t, mask = cv2.recoverPose(E, points2, points1, focal=focal, pp=pp, mask=mask)
    
    prev_image = img_2
    prev_features = points2
    

    cv2.namedWindow("Road facing Egomotion camera", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Trajectory", cv2.WINDOW_AUTOSIZE)
    
    # Create trajectory image
    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Process frames
    for num_frame in range(2, MAX_FRAME):
        # Read current frame
        filename = os.path.join(dataset_path, f"image_0/{num_frame:06d}.png")
        curr_image_c = cv2.imread(filename)
        
        if curr_image_c is None:
            print(f"Could not read frame {num_frame}")
            break
            
        # Convert to grayscale
        curr_image = cv2.cvtColor(curr_image_c, cv2.COLOR_BGR2GRAY)
        
        # Track features
        curr_features = np.array([])
        prev_features, curr_features, status = featureTracking(prev_image, curr_image, prev_features, curr_features)
        
        # Find essential matrix and recover pose
        E, mask = cv2.findEssentialMat(curr_features, prev_features, focal, pp, cv2.RANSAC, 0.999, 1.0)
        _, R, t, mask = cv2.recoverPose(E, curr_features, prev_features, focal=focal, pp=pp, mask=mask)
        
        # Convert feature points format
        prevPts = np.zeros((2, len(prev_features)))
        currPts = np.zeros((2, len(curr_features)))
        
        for i in range(len(prev_features)):
            prevPts[0, i] = prev_features[i][0]
            prevPts[1, i] = prev_features[i][1]
            
            currPts[0, i] = curr_features[i][0]
            currPts[1, i] = curr_features[i][1]
        
        # Get scale
        scale = getAbsoluteScale(num_frame)
        
        # Update pose if conditions are met
        if scale > 0.1 and t[2, 0] > t[0, 0] and t[2, 0] > t[1, 0]:
            t_f = t_f + scale * (R_f @ t)  # t_final = t_previous + scale * (R_previous * t_current)
            R_f = R @ R_f  # R_final = R_current * R_previous
        
        # Redetect features if needed
        if len(prev_features) < MIN_NUM_FEAT:
            prev_features = featureDetection(prev_image)
            prev_features, curr_features, status = featureTracking(prev_image, curr_image, prev_features, curr_features)
        
        # Update previous frame
        prev_image = curr_image.copy()
        prev_features = curr_features
        
        # Visualization
        x = int(t_f[0, 0]) + 300  # Offset for easier visualization
        y = int(-1 * t_f[2, 0]) + 500  # -1 inversion and offset for easier visualization
        cv2.circle(traj, (x, y), 1, (255, 0, 0), 2)
        
        # Display coordinates
        cv2.rectangle(traj, (10, 30), (550, 50), (0, 0, 0), cv2.FILLED)
        text = f"Coordinates: x = {t_f[0, 0]:.2f}m y = {t_f[1, 0]:.2f}m z = {t_f[2, 0]:.2f}m"
        cv2.putText(traj, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        
        # Show images
        cv2.imshow("Road facing Egomotion camera", curr_image_c)
        cv2.imshow("Trajectory", traj)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    
    cv2.destroyAllWindows()
    return 0


VisualOdometry.run = run


if __name__ == "__main__":
    vo = VisualOdometry()
    dataset_path = "monocular_vo/kitti_dataset/data_odometry_gray/dataset/sequences/00"  
    result = vo.run(dataset_path)
    if result == 0:
        print("Visual Odometry completed successfully")
    else:
        print("Visual Odometry failed")