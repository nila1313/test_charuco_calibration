import cv2
import glob
import os

os.makedirs("debug_aruco", exist_ok=True)

image_paths = sorted(glob.glob("selected_frames/*.jpg"))

dictionaries = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

results = []

for dict_name, dict_id in dictionaries.items():
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 99
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0

    detector = cv2.aruco.ArucoDetector(dictionary, params)

    total = 0
    good_frames = 0

    for i, path in enumerate(image_paths[:50]):
        img = cv2.imread(path)

        # resize bigger to help detection
        img_big = cv2.resize(img, None, fx=2, fy=2)

        gray = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)

        # improve contrast
        gray = cv2.equalizeHist(gray)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            total += len(ids)
            good_frames += 1

            debug = img_big.copy()
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)
            cv2.imwrite(f"debug_aruco/{dict_name}_frame_{i:04d}.jpg", debug)

    results.append((good_frames, total, dict_name))

results = sorted(results, reverse=True)

print("\nAruco detection results:")
for r in results:
    print(r)