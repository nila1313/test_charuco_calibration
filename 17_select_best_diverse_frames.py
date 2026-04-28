import cv2
import glob
import os
import numpy as np
import shutil

input_dir = "all_good_charuco_frames"
output_dir = "best_diverse_frames"

os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(f"{input_dir}/*.jpg"))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard(
    (7, 7),
    1.0,
    0.8,
    dictionary
)

params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

selected = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        continue

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    if charuco_ids is None:
        continue

    n = len(charuco_ids)

    if n < 8:
        continue

    # compute center of board
    center = np.mean(charuco_corners.reshape(-1, 2), axis=0)

    selected.append((path, n, center))

# sort by number of corners (descending)
selected.sort(key=lambda x: -x[1])

final = []
used_centers = []

for path, n, center in selected:
    # avoid similar viewpoints
    keep = True
    for c in used_centers:
        if np.linalg.norm(center - c) < 50:
            keep = False
            break

    if keep:
        final.append((path, n))
        used_centers.append(center)

    if len(final) >= 60:
        break

# copy selected frames
for path, n in final:
    name = os.path.basename(path)
    shutil.copy(path, f"{output_dir}/{name}")
    print(name, "corners:", n)

print("\nSelected best frames:", len(final))