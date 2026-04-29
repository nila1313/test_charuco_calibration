import cv2
import numpy as np
import glob
import os

# -----------------------------
# Load best calibration result
# -----------------------------
data = np.load("charuco_calibration_result_best_robust.npz")

K = data["camera_matrix"]
D = data["dist_coeffs"]

print("Loaded calibration:")
print("K:\n", K)
print("D:\n", D)

# -----------------------------
# Input / output folders
# -----------------------------
input_folder = "all_good_charuco_frames"
output_folder = "undistorted_best_robust"

os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob.glob(f"{input_folder}/*.jpg"))

# Use only first 30 frames for visual check
image_paths = image_paths[:30]

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    h, w = img.shape[:2]

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        D,
        (w, h),
        alpha=0.3,
        newImgSize=(w, h)
    )

    undistorted = cv2.undistort(img, K, D, None, new_K)

    # Side-by-side comparison
    combined = np.hstack((img, undistorted))

    name = os.path.basename(path)
    cv2.imwrite(f"{output_folder}/{name}", combined)

print(f"Saved before/after images in: {output_folder}")