import cv2
import glob
import numpy as np

image_folder = "all_good_charuco_frames"

# gamma
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma = 1.8

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(gray, table)

# board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard((7, 7), 1.0, 0.8, dictionary)

image_paths = sorted(glob.glob(f"{image_folder}/*.jpg"))

# ----------------------------
# PARAMETER SETS TO TEST
# ----------------------------
configs = {
    "default": {},
    
    "high_refine": {
        "cornerRefinementMethod": cv2.aruco.CORNER_REFINE_SUBPIX,
        "cornerRefinementWinSize": 5,
        "cornerRefinementMaxIterations": 100,
        "cornerRefinementMinAccuracy": 0.0001
    },

    "very_strict": {
        "cornerRefinementMethod": cv2.aruco.CORNER_REFINE_SUBPIX,
        "cornerRefinementWinSize": 7,
        "cornerRefinementMaxIterations": 150,
        "cornerRefinementMinAccuracy": 0.00001
    },

    "fast_relaxed": {
        "cornerRefinementMethod": cv2.aruco.CORNER_REFINE_SUBPIX,
        "cornerRefinementWinSize": 3,
        "cornerRefinementMaxIterations": 30,
        "cornerRefinementMinAccuracy": 0.01
    }
}

# ----------------------------
# TEST LOOP
# ----------------------------
for name, cfg in configs.items():
    params = cv2.aruco.DetectorParameters()

    for k, v in cfg.items():
        setattr(params, k, v)

    detector = cv2.aruco.ArucoDetector(dictionary, params)

    total_markers = 0
    total_charuco = 0
    max_charuco = 0
    valid_frames = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        gray = preprocess(img)

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if marker_ids is None:
            continue

        total_markers += len(marker_ids)

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board
        )

        if charuco_ids is None:
            continue

        n = len(charuco_ids)

        total_charuco += n
        max_charuco = max(max_charuco, n)
        valid_frames += 1

    if valid_frames > 0:
        print(
            f"{name} | "
            f"avg markers: {total_markers/valid_frames:.2f} | "
            f"avg charuco: {total_charuco/valid_frames:.2f} | "
            f"max charuco: {max_charuco}"
        )