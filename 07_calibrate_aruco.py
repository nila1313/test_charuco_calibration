import cv2
import glob
import numpy as np

image_paths = sorted(glob.glob("best_frames/*.jpg"))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Your board seems to contain marker IDs 0–23 = 24 markers
# Try 6 columns × 4 rows first
board = cv2.aruco.GridBoard(
    size=(6, 4),
    markerLength=1.0,
    markerSeparation=0.3,
    dictionary=dictionary
)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

all_corners = []
all_ids = []
counter = []

image_size = None

detector = cv2.aruco.ArucoDetector(dictionary, params)

for path in image_paths:
    img = cv2.imread(path)
    image_size = img.shape[1], img.shape[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        continue

    ids_flat = ids.flatten()
    valid = [i for i, v in enumerate(ids_flat) if 0 <= v <= 23]

    if len(valid) >= 6:
        all_corners.extend([corners[i] for i in valid])
        all_ids.extend(ids[valid])
        counter.append(len(valid))

all_ids = np.array(all_ids, dtype=np.int32)

print("Frames used:", len(counter))
print("Markers used:", len(all_ids))
print("Image size:", image_size)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
    all_corners,
    all_ids,
    np.array(counter),
    board,
    image_size,
    None,
    None
)

print("Reprojection error:", ret)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

np.savez(
    "calibration_result.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print("Saved calibration_result.npz")