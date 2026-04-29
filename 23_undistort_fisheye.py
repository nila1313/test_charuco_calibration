import cv2
import numpy as np
import glob
import os

data = np.load("charuco_fisheye_calibration_result.npz")

K = data["camera_matrix"]
D = data["dist_coeffs"]

print("Loaded fisheye calibration:")
print("K:\n", K)
print("D:\n", D)

input_folder = "all_good_charuco_frames"
output_folder = "undistorted_fisheye"

os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob.glob(f"{input_folder}/*.jpg"))

# test first 30 images
image_paths = image_paths[:30]

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    h, w = img.shape[:2]

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.3
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    combined = np.hstack((img, undistorted))

    name = os.path.basename(path)
    cv2.imwrite(f"{output_folder}/{name}", combined)

print("Saved fisheye undistorted images.")