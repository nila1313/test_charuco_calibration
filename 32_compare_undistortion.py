import cv2
import numpy as np
import glob
import os

# -----------------------------
# Load calibrations
# -----------------------------
old = np.load("charuco_omnidir_calibration_result.npz")
new = np.load("charuco_omnidir_gamma18.npz")

K_old = old["camera_matrix"]
xi_old = np.asarray(old["xi"], dtype=np.float64).reshape(1, 1)
D_old = old["dist_coeffs"]

K_new = new["camera_matrix"]
xi_new = np.asarray(new["xi"], dtype=np.float64).reshape(1, 1)
D_new = new["dist_coeffs"]

print("Loaded old calibration:")
print("K old:\n", K_old)
print("xi old:", xi_old)
print("D old:\n", D_old)

print("\nLoaded gamma calibration:")
print("K new:\n", K_new)
print("xi new:", xi_new)
print("D new:\n", D_new)

# -----------------------------
# Input / output
# -----------------------------
input_folder = "all_good_charuco_frames"
output_folder = "compare_undistortion"

os.makedirs(output_folder, exist_ok=True)

# clear old outputs
for f in glob.glob(f"{output_folder}/*.jpg"):
    os.remove(f)

image_paths = sorted(glob.glob(f"{input_folder}/*.jpg"))

# choose representative frames
step = max(1, len(image_paths) // 10)
image_paths = image_paths[::step][:10]

# -----------------------------
# Compare undistortion
# -----------------------------
for i, path in enumerate(image_paths):
    img = cv2.imread(path)

    if img is None:
        continue

    h, w = img.shape[:2]
    R = np.eye(3)

    # OLD omnidir undistortion
    map1_old, map2_old = cv2.omnidir.initUndistortRectifyMap(
        K_old,
        D_old,
        xi_old,
        R,
        K_old,
        (w, h),
        cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    undist_old = cv2.remap(
        img,
        map1_old,
        map2_old,
        interpolation=cv2.INTER_LINEAR
    )

    # NEW gamma omnidir undistortion
    map1_new, map2_new = cv2.omnidir.initUndistortRectifyMap(
        K_new,
        D_new,
        xi_new,
        R,
        K_new,
        (w, h),
        cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    undist_new = cv2.remap(
        img,
        map1_new,
        map2_new,
        interpolation=cv2.INTER_LINEAR
    )

    combined = np.hstack([
        img,
        undist_old,
        undist_new
    ])

    cv2.putText(
        combined,
        "Original",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        combined,
        "Old omnidir RMS ~2.12",
        (w + 50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        combined,
        "Gamma 1.8 RMS ~1.75",
        (2 * w + 50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    out_name = f"compare_{i}.jpg"
    cv2.imwrite(os.path.join(output_folder, out_name), combined)

print("Saved comparison images in:", output_folder)