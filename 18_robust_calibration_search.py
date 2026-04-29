import cv2
import glob
import numpy as np
import os

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_for_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)

    return sharp


# -----------------------------
# Collect ChArUco detections
# -----------------------------
def collect_data(input_folder, min_corners=8, max_corners=20):
    image_paths = sorted(glob.glob(f"{input_folder}/*.jpg"))

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

        if not (min_corners <= n <= max_corners):
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


# -----------------------------
# Calibrate
# -----------------------------
def calibrate(frame_data, board, image_size, flags=0):
    corners = [f["corners"] for f in frame_data]
    ids = [f["ids"] for f in frame_data]

    error, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        corners,
        ids,
        board,
        image_size,
        None,
        None,
        flags=flags
    )

    return error, K, D, rvecs, tvecs


# -----------------------------
# Per-frame reprojection error
# -----------------------------
def compute_frame_errors(frame_data, board, K, D, rvecs, tvecs):
    obj_points_all = board.getChessboardCorners()
    errors = []

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
        err = np.linalg.norm(img_points - projected, axis=1)

        errors.append({
            "index": i,
            "path": f["path"],
            "mean_error": float(np.mean(err)),
            "max_error": float(np.max(err)),
            "n": f["n"]
        })

    return errors


# -----------------------------
# Iterative outlier removal
# -----------------------------
def robust_prune(frame_data, board, image_size, flags=0, max_rounds=10):
    current = frame_data[:]

    best_error, best_K, best_D, best_rvecs, best_tvecs = calibrate(
        current, board, image_size, flags
    )

    best_data = current[:]

    print("Initial error:", best_error, "frames:", len(current))

    for round_id in range(max_rounds):
        error, K, D, rvecs, tvecs = calibrate(current, board, image_size, flags)
        frame_errors = compute_frame_errors(current, board, K, D, rvecs, tvecs)

        frame_errors = sorted(frame_errors, key=lambda x: x["mean_error"], reverse=True)

        # remove only 5% at a time, not 20%
        remove_count = max(1, int(len(current) * 0.05))
        bad_indices = set(e["index"] for e in frame_errors[:remove_count])

        candidate = [
            f for i, f in enumerate(current)
            if i not in bad_indices
        ]

        if len(candidate) < 20:
            print("Stopping: too few frames left.")
            break

        new_error, new_K, new_D, new_rvecs, new_tvecs = calibrate(
            candidate, board, image_size, flags
        )

        print(
            f"Round {round_id + 1}: "
            f"removed {remove_count}, "
            f"frames {len(candidate)}, "
            f"error {new_error}"
        )

        # only accept if error improves
        if new_error < best_error:
            best_error = new_error
            best_K = new_K
            best_D = new_D
            best_rvecs = new_rvecs
            best_tvecs = new_tvecs
            best_data = candidate[:]
            current = candidate[:]
        else:
            print("No improvement. Stop pruning.")
            break

    return best_error, best_K, best_D, best_data


# -----------------------------
# Main experiment
# -----------------------------
experiments = [
    {
        "name": "standard_8_20",
        "min_corners": 8,
        "max_corners": 20,
        "flags": 0
    },
    {
        "name": "standard_9_20",
        "min_corners": 9,
        "max_corners": 20,
        "flags": 0
    },
    {
        "name": "standard_10_20",
        "min_corners": 10,
        "max_corners": 20,
        "flags": 0
    },
    {
        "name": "rational_8_20",
        "min_corners": 8,
        "max_corners": 20,
        "flags": cv2.CALIB_RATIONAL_MODEL
    },
]

best_global = None

for exp in experiments:
    print("\n==============================")
    print("Experiment:", exp["name"])
    print("==============================")

    frame_data, board, image_size = collect_data(
        "all_good_charuco_frames",
        min_corners=exp["min_corners"],
        max_corners=exp["max_corners"]
    )

    print("Collected frames:", len(frame_data))

    if len(frame_data) < 20:
        print("Skipping: too few frames.")
        continue

    try:
        error, K, D, used_data = robust_prune(
            frame_data,
            board,
            image_size,
            flags=exp["flags"],
            max_rounds=10
        )
    except Exception as e:
        print("Experiment failed:", e)
        continue

    print("\nFinal for", exp["name"])
    print("Frames used:", len(used_data))
    print("Error:", error)
    print("K:\n", K)
    print("D:\n", D)

    if best_global is None or error < best_global["error"]:
        best_global = {
            "name": exp["name"],
            "error": error,
            "K": K,
            "D": D,
            "frames": len(used_data)
        }

if best_global is not None:
    print("\n==============================")
    print("BEST RESULT")
    print("==============================")
    print("Experiment:", best_global["name"])
    print("Frames:", best_global["frames"])
    print("Error:", best_global["error"])
    print("K:\n", best_global["K"])
    print("D:\n", best_global["D"])

    np.savez(
        "charuco_calibration_result_best_robust.npz",
        camera_matrix=best_global["K"],
        dist_coeffs=best_global["D"]
    )

    print("\nSaved charuco_calibration_result_best_robust.npz")
else:
    print("No successful calibration.")