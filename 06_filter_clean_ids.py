import cv2
import glob
import os
import shutil

os.makedirs("best_frames", exist_ok=True)
os.makedirs("debug_best_frames", exist_ok=True)

image_paths = sorted(glob.glob("frames/*.jpg"))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.minMarkerPerimeterRate = 0.01
params.maxMarkerPerimeterRate = 4.0

detector = cv2.aruco.ArucoDetector(dictionary, params)

kept = 0

for path in image_paths:
    img = cv2.imread(path)
    big = cv2.resize(img, None, fx=2, fy=2)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        continue

    ids_flat = ids.flatten()
    valid = [i for i, v in enumerate(ids_flat) if 0 <= v <= 23]

    if len(valid) >= 8:
        kept += 1
        name = os.path.basename(path)
        shutil.copy(path, f"best_frames/{name}")

        good_corners = [corners[i] for i in valid]
        good_ids = ids[valid]

        debug = big.copy()
        cv2.aruco.drawDetectedMarkers(debug, good_corners, good_ids)
        cv2.imwrite(f"debug_best_frames/{name}", debug)

        print(name, "valid markers:", len(valid), "ids:", good_ids.flatten().tolist())

print("Best frames kept:", kept)