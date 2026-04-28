import cv2
import glob
import numpy as np
import os

def preprocess_for_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)

    return sharp


def collect_charuco_data():
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

    detector = cv2.aruco.ArucoDetector(dictionary, params)

    frame_data = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        image_size = (img.shape[1], img.shape[0])
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

        if not (8 <= n <= 20):
            continue

        if retval < 0.5:
            continue

        frame_data.append({
            "path": path,
            "corners": charuco_corners,
            "ids": charuco_ids,
            "n": n
        })

    return frame_data, board, image_size


def calibrate(frame_data, board, image_size):
    corners = [f["corners"] for f in frame_data]
    ids = [f["ids"] for f in frame_data]

    error, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        corners,
        ids,
        board,
        image_size,
        None,
        None
    )

    return error, K, D, rvecs, tvecs


def compute_frame_errors(frame_data, board, K, D, rvecs, tvecs):
    frame_errors = []

    # 3D ChArUco board corner coordinates
    obj_points_all = board.getChessboardCorners()

    for i, f in enumerate(frame_data):
        ids = f["ids"].flatten()
        img_points = f["corners"].reshape(-1, 2)

        obj_points = obj_points_all[ids].astype(np.float32)

        projected, _ = cv2.projectPoints(
            obj_points,
            rvecs[i],
            tvecs[i],
            K,
            D
        )

        projected = projected.reshape(-1, 2)

        errors = np.linalg.norm(img_points - projected, axis=1)
        mean_error = float(np.mean(errors))

        frame_errors.append({
            "index": i,
            "path": f["path"],
            "mean_error": mean_error,
            "n": f["n"]
        })

    return frame_errors


frame_data, board, image_size = collect_charuco_data()

print("Initial frames:", len(frame_data))
print("Image size:", image_size)

error1, K1, D1, rvecs1, tvecs1 = calibrate(frame_data, board, image_size)

print("\nInitial calibration error:", error1)

frame_errors = compute_frame_errors(frame_data, board, K1, D1, rvecs1, tvecs1)

frame_errors_sorted = sorted(frame_errors, key=lambda x: x["mean_error"], reverse=True)

print("\nWorst frames:")
for item in frame_errors_sorted[:10]:
    print(
        os.path.basename(item["path"]),
        "mean_error:",
        round(item["mean_error"], 3),
        "corners:",
        item["n"]
    )

# Remove worst 20% frames
remove_count = max(1, int(len(frame_data) * 0.20))
bad_indices = set(item["index"] for item in frame_errors_sorted[:remove_count])

cleaned_frame_data = [
    f for i, f in enumerate(frame_data)
    if i not in bad_indices
]

print("\nRemoved frames:", remove_count)
print("Remaining frames:", len(cleaned_frame_data))

error2, K2, D2, rvecs2, tvecs2 = calibrate(cleaned_frame_data, board, image_size)

print("\nFinal calibration error:", error2)
print("Camera matrix:\n", K2)
print("Distortion coefficients:\n", D2)

np.savez(
    "charuco_calibration_result_outlier_removed.npz",
    camera_matrix=K2,
    dist_coeffs=D2
)

print("\nSaved charuco_calibration_result_outlier_removed.npz")