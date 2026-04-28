import cv2
import glob
import os
import shutil

def preprocess_for_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)

    return sharp


input_dir = "all_video_frames"
output_dir = "all_good_charuco_frames"
debug_dir = "debug_all_good_charuco"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

image_paths = sorted(glob.glob(f"{input_dir}/*.jpg"))

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

detector = cv2.aruco.ArucoDetector(dictionary, params)

kept = 0

for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is None:
        continue

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

    # same filter as best calibration
    if not (8 <= n <= 20):
        continue

    if retval < 0.5:
        continue

    name = os.path.basename(path)
    shutil.copy(path, f"{output_dir}/{name}")

    debug = img.copy()
    cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
    cv2.imwrite(f"{debug_dir}/{name}", debug)

    kept += 1

    if kept % 50 == 0:
        print("Kept:", kept, "latest:", name, "corners:", n)

print("\nTotal frames checked:", len(image_paths))
print("Good ChArUco frames kept:", kept)