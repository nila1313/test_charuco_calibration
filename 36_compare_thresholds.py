import cv2
import numpy as np
import glob
import os

# -----------------------------
# Load calibrations
# -----------------------------
old = np.load("charuco_omnidir_gamma18.npz")
best = np.load("charuco_omnidir_gamma18_best_threshold.npz")

K_old = old["camera_matrix"]
xi_old = np.asarray(old["xi"], dtype=np.float64).reshape(1, 1)
D_old = old["dist_coeffs"]

K_best = best["camera_matrix"]
xi_best = np.asarray(best["xi"], dtype=np.float64).reshape(1, 1)
D_best = best["dist_coeffs"]

# -----------------------------
# Images
# -----------------------------
image_paths = sorted(glob.glob("all_good_charuco_frames/*.jpg"))

step = max(1, len(image_paths) // 10)
image_paths = image_paths[::step][:10]

output_folder = "compare_thresholds"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Compare
# -----------------------------
for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    R = np.eye(3)

    # OLD (9–20)
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

    undist_old = cv2.remap(img, map1_old, map2_old, cv2.INTER_LINEAR)

    # BEST (12–20)
    map1_best, map2_best = cv2.omnidir.initUndistortRectifyMap(
        K_best,
        D_best,
        xi_best,
        R,
        K_best,
        (w, h),
        cv2.CV_32FC1,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    undist_best = cv2.remap(img, map1_best, map2_best, cv2.INTER_LINEAR)

    combined = np.hstack([img, undist_old, undist_best])

    cv2.putText(combined, "Original", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(combined, "Gamma 9–20 (~1.75)", (w+50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(combined, "Gamma 12–20 (~1.70)", (2*w+50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imwrite(f"{output_folder}/compare_{i}.jpg", combined)

print("Saved comparison in:", output_folder)