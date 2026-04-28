import cv2
import glob
import os

os.makedirs("debug_selected_aruco", exist_ok=True)

image_paths = sorted(glob.glob("selected_frames/*.jpg"))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

params = cv2.aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 99
params.adaptiveThreshWinSizeStep = 10
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.minMarkerPerimeterRate = 0.01
params.maxMarkerPerimeterRate = 4.0

detector = cv2.aruco.ArucoDetector(dictionary, params)

for path in image_paths:
    img = cv2.imread(path)
    big = cv2.resize(img, None, fx=2, fy=2)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    corners, ids, rejected = detector.detectMarkers(gray)

    debug = big.copy()

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
        print(path, "markers:", len(ids), "ids:", ids.flatten().tolist())
    else:
        print(path, "markers: 0")

    name = os.path.basename(path)
    cv2.imwrite(f"debug_selected_aruco/{name}", debug)