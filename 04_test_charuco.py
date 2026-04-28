import cv2
import glob
import os

os.makedirs("debug_charuco", exist_ok=True)

image_paths = sorted(glob.glob("selected_frames/*.jpg"))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard(
    (7, 7),
    1.0,
    0.7,
    dictionary
)

charuco_detector = cv2.aruco.CharucoDetector(board)

good = 0
total_corners = 0

for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_ids is not None and len(charuco_ids) >= 6:
        good += 1
        total_corners += len(charuco_ids)

        debug = img.copy()
        cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
        cv2.imwrite(f"debug_charuco/charuco_{i:04d}.jpg", debug)

        print(path, "corners:", len(charuco_ids))
    else:
        print(path, "FAILED")

print("\nGood frames:", good)
print("Total corners:", total_corners)