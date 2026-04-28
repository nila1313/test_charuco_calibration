import cv2
import glob
import numpy as np
import os

def preprocess_for_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Local contrast improvement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Softer sharpening than before
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)

    return sharp


image_paths = sorted(glob.glob("good_frames/*.jpg"))

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

params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 151
params.adaptiveThreshWinSizeStep = 10

params.minMarkerPerimeterRate = 0.005
params.maxMarkerPerimeterRate = 4.0
params.polygonalApproxAccuracyRate = 0.03
params.minCornerDistanceRate = 0.03
params.minDistanceToBorder = 2

aruco_detector = cv2.aruco.ArucoDetector(dictionary, params)

all_charuco_corners = []
all_charuco_ids = []
image_size = None

os.makedirs("debug_charuco_filtered", exist_ok=True)

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    image_size = (img.shape[1], img.shape[0])
    gray = preprocess_for_detection(img)

    marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(gray)

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

    n_corners = len(charuco_ids)

    # Keep medium/strong detections only
    if not (8 <= n_corners <= 20):
        continue

    # Reject weak interpolation
    if retval < 0.5:
        continue

    all_charuco_corners.append(charuco_corners)
    all_charuco_ids.append(charuco_ids)

    debug = img.copy()
    cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)

    name = os.path.basename(path)
    cv2.imwrite(f"debug_charuco_filtered/{name}", debug)

    print(name, "charuco corners:", n_corners, "retval:", retval)

print("\nFrames used:", len(all_charuco_corners))
print("Image size:", image_size)

if len(all_charuco_corners) < 5:
    print("Not enough good ChArUco frames.")
    exit()

error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners,
    all_charuco_ids,
    board,
    image_size,
    None,
    None
)

print("\nReprojection error:", error)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

np.savez(
    "charuco_calibration_result_filtered.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print("\nSaved charuco_calibration_result_filtered.npz")