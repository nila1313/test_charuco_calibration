import cv2
import glob
import os
import shutil
import numpy as np

input_folder = "all_good_charuco_frames"
output_folder = "omnidir_diverse_frames"

os.makedirs(output_folder, exist_ok=True)

# clear old output
for f in glob.glob(f"{output_folder}/*.jpg"):
    os.remove(f)

def preprocess_for_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)
    return sharp

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard(
    (7, 7),
    1.0,
    0.8,
    dictionary
)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 50
params.cornerRefinementMinAccuracy = 0.001

detector = cv2.aruco.ArucoDetector(dictionary, params)

image_paths = sorted(glob.glob(f"{input_folder}/*.jpg"))

candidates = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    h, w = img.shape[:2]
    gray = preprocess_for_detection(img)

    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) < 4:
        continue

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board
    )

    if charuco_ids is None:
        continue

    n = len(charuco_ids)

    if not (9 <= n <= 20):
        continue

    pts = charuco_corners.reshape(-1, 2)

    cx = np.mean(pts[:, 0]) / w
    cy = np.mean(pts[:, 1]) / h

    spread_x = (np.max(pts[:, 0]) - np.min(pts[:, 0])) / w
    spread_y = (np.max(pts[:, 1]) - np.min(pts[:, 1])) / h

    area_score = spread_x * spread_y

    # reject tiny detections
    if area_score < 0.01:
        continue

    candidates.append({
        "path": path,
        "corners": n,
        "cx": cx,
        "cy": cy,
        "area": area_score
    })

print("Candidates:", len(candidates))

# Divide image into grid cells
grid_rows = 4
grid_cols = 4
max_per_cell = 20

selected = []

for r in range(grid_rows):
    for c in range(grid_cols):
        x_min = c / grid_cols
        x_max = (c + 1) / grid_cols
        y_min = r / grid_rows
        y_max = (r + 1) / grid_rows

        cell_items = [
            item for item in candidates
            if x_min <= item["cx"] < x_max and y_min <= item["cy"] < y_max
        ]

        # prefer larger board coverage and more corners
        cell_items = sorted(
            cell_items,
            key=lambda x: (x["area"], x["corners"]),
            reverse=True
        )

        selected.extend(cell_items[:max_per_cell])

print("Selected diverse frames:", len(selected))

for item in selected:
    src = item["path"]
    dst = os.path.join(output_folder, os.path.basename(src))
    shutil.copy(src, dst)

print("Saved selected frames to:", output_folder)