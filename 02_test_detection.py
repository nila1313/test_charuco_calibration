import cv2
import glob
import os

os.makedirs("debug_detection", exist_ok=True)

image_paths = sorted(glob.glob("frames/*.jpg"))

dictionaries = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}

# try possible board sizes
board_sizes = [
    (8, 6),
    (9, 6),
    (10, 7),
    (7, 5),
    (5, 7),
]

best = []

for dict_name, dict_id in dictionaries.items():
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    for board_size in board_sizes:
        board = cv2.aruco.CharucoBoard(
            board_size,
            1.0,   # square length
            0.7,   # marker length
            dictionary
        )

        total_corners = 0
        good_frames = 0

        for path in image_paths[:30]:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

            corners, ids, _ = detector.detectMarkers(gray)

            if ids is None:
                continue

            charuco_detector = cv2.aruco.CharucoDetector(board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

            if charuco_ids is not None and len(charuco_ids) > 4:
                total_corners += len(charuco_ids)
                good_frames += 1

        best.append((good_frames, total_corners, dict_name, board_size))

best = sorted(best, reverse=True)

print("\nBest results:")
for item in best[:10]:
    print(item)