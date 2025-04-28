#Andrew Aquino Task 4

import cv2
import numpy as np
import os

def getCalibrationData(dataset_path):
    """
    Load calibration values from KITTI's calibration files
    """
    calibration_file_path = os.path.join(dataset_path, "calib.txt")
    focal = 0.0
    pp = (0.0, 0.0)
    
    if os.path.exists(calibration_file_path):
        with open(calibration_file_path, 'r') as myfile:
            for line in myfile:
                results = line.strip().split()
                focal = float(results[1])
                pp_x = float(results[3])
                pp_y = float(results[7])
                pp = (pp_x, pp_y)
                break
    
    return focal, pp

def featureTracking(img_1, img_2, points1, points2):
    """
    Track features between two frames using KLT tracker
    """
    window_size = (21, 21)
    max_level = 3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    min_eig_threshold = 0.001
    

    points2, status, err = cv2.calcOpticalFlowPyrLK(
        img_1, img_2, np.array(points1, dtype=np.float32), 
        None, winSize=window_size, maxLevel=max_level, 
        criteria=criteria, minEigThreshold=min_eig_threshold
    )
    
    # Filter out invalid points
    valid_points1 = []
    valid_points2 = []
    
    for i, (pt, stat) in enumerate(zip(points2, status)):
        if stat == 1 and pt[0] >= 0 and pt[1] >= 0:
            valid_points1.append(points1[i])
            valid_points2.append(pt)
    
    return np.array(valid_points1, dtype=np.float32), np.array(valid_points2, dtype=np.float32), status

def featureDetection(img_1):
    """
    Detect features using FAST detector
    """
    fast_threshold = 20
    non_max_suppression = True
    
    # Detect FAST features
    detector = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=non_max_suppression)
    keypoints = detector.detect(img_1, None)
    
    # Convert KeyPoint objects to a list of points
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    
    return points
