import cv2
import glob
import os
import numpy as np

image_paths = sorted(glob.glob("good_frames/*.jpg"))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board_sizes = [
    (7, 7),
    (8, 7),
    (9, 7),
    (8, 8),
    (9, 8),
    (10, 7),
]

marker_ratios = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 99
params.adaptiveThreshWinSizeStep = 10
params.minMarkerPerimeterRate = 0.005
params.maxMarkerPerimeterRate = 4.0

detector = cv2.aruco.ArucoDetector(dictionary, params)

results = []

for board_size in board_sizes:
    for ratio in marker_ratios:
        board = cv2.aruco.CharucoBoard(
            board_size,
            1.0,
            ratio,
            dictionary
        )

        good = 0
        total_corners = 0
        max_corners = 0

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            marker_corners, marker_ids, _ = detector.detectMarkers(gray)

            if marker_ids is None or len(marker_ids) < 4:
                continue

            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board
            )

            if charuco_ids is not None and len(charuco_ids) >= 4:
                good += 1
                total_corners += len(charuco_ids)
                max_corners = max(max_corners, len(charuco_ids))

        results.append((good, total_corners, max_corners, board_size, ratio))

results = sorted(results, reverse=True)

print("Best ChArUco settings:")
for r in results[:20]:
    print(r)