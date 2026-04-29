import cv2
import numpy as np
import glob

# -----------------------------
# Settings
# -----------------------------
image_folder = "all_good_charuco_frames"
gamma_value = 1.8
min_corners = 9
max_corners = 20
remove_ratio = 0.05
max_rounds = 10

# -----------------------------
# Gamma preprocessing
# -----------------------------
def preprocess_gamma(img, gamma=1.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(gray, table)

# -----------------------------
# ChArUco setup
# -----------------------------
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard(
    (7, 7),
    1.0,
    0.8,
    dictionary
)

all_board_corners = board.getChessboardCorners()

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 80
params.cornerRefinementMinAccuracy = 0.0005

detector = cv2.aruco.ArucoDetector(dictionary, params)

# -----------------------------
# Collect points
# -----------------------------
image_paths = sorted(glob.glob(f"{image_folder}/*.jpg"))

objpoints = []
imgpoints = []
used_paths = []
image_size = None

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    image_size = (img.shape[1], img.shape[0])
    gray = preprocess_gamma(img, gamma_value)

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

    ids = charuco_ids.flatten()

    obj = all_board_corners[ids].astype(np.float64)
    imgp = charuco_corners.reshape(-1, 2).astype(np.float64)

    objpoints.append(obj.reshape(-1, 1, 3))
    imgpoints.append(imgp.reshape(-1, 1, 2))
    used_paths.append(path)

print("Images found:", len(image_paths))
print("Collected frames:", len(objpoints))

if len(objpoints) < 10:
    print("Not enough detected frames.")
    exit()

# -----------------------------
# Calibration function
# -----------------------------
def calibrate_omnidir(objpoints, imgpoints, image_size):
    w, h = image_size

    K = np.array([
        [w / 2, 0, w / 2],
        [0, w / 2, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    xi = np.array([[1.0]], dtype=np.float64)
    D = np.zeros((1, 4), dtype=np.float64)

    flags = (
        cv2.omnidir.CALIB_USE_GUESS
        + cv2.omnidir.CALIB_FIX_SKEW
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-8
    )

    rms, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        xi,
        D,
        flags,
        criteria
    )

    return rms, K, xi, D, rvecs, tvecs

# -----------------------------
# Compute frame errors
# -----------------------------
def compute_errors(objpoints, imgpoints, rvecs, tvecs, K, xi, D):
    errors = []
    xi_scalar = float(np.asarray(xi).reshape(-1)[0])

    usable = min(len(objpoints), len(rvecs), len(tvecs))

    for i in range(usable):
        projected, _ = cv2.omnidir.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            K,
            xi_scalar,
            D
        )

        projected = projected.reshape(-1, 2)
        detected = imgpoints[i].reshape(-1, 2)

        err = np.linalg.norm(projected - detected, axis=1).mean()
        errors.append({"index": i, "error": err})

    return errors

# -----------------------------
# Initial calibration
# -----------------------------
best_rms, best_K, best_xi, best_D, best_rvecs, best_tvecs = calibrate_omnidir(
    objpoints,
    imgpoints,
    image_size
)

best_obj = objpoints[:]
best_img = imgpoints[:]

print("Initial RMS:", best_rms)

# -----------------------------
# Robust pruning
# -----------------------------
for round_id in range(max_rounds):
    frame_errors = compute_errors(
        best_obj,
        best_img,
        best_rvecs,
        best_tvecs,
        best_K,
        best_xi,
        best_D
    )

    frame_errors = sorted(frame_errors, key=lambda x: x["error"], reverse=True)

    remove_count = max(1, int(len(frame_errors) * remove_ratio))
    bad_indices = set(e["index"] for e in frame_errors[:remove_count])

    candidate_obj = [
        p for i, p in enumerate(best_obj)
        if i not in bad_indices
    ]

    candidate_img = [
        p for i, p in enumerate(best_img)
        if i not in bad_indices
    ]

    if len(candidate_obj) < 30:
        print("Too few frames left. Stop.")
        break

    new_rms, new_K, new_xi, new_D, new_rvecs, new_tvecs = calibrate_omnidir(
        candidate_obj,
        candidate_img,
        image_size
    )

    print(
        f"Round {round_id + 1}: "
        f"removed {remove_count}, "
        f"frames {len(candidate_obj)}, "
        f"RMS {new_rms}"
    )

    if new_rms < best_rms:
        best_rms = new_rms
        best_K = new_K
        best_xi = new_xi
        best_D = new_D
        best_rvecs = new_rvecs
        best_tvecs = new_tvecs
        best_obj = candidate_obj[:]
        best_img = candidate_img[:]
    else:
        print("No improvement. Stop.")
        break

# -----------------------------
# Save result
# -----------------------------
print("\nFINAL RESULT")
print("Frames:", len(best_obj))
print("RMS:", best_rms)
print("K:\n", best_K)
print("xi:\n", best_xi)
print("D:\n", best_D)

np.savez(
    "charuco_omnidir_gamma18_robust.npz",
    camera_matrix=best_K,
    xi=best_xi,
    dist_coeffs=best_D
)

print("\nSaved charuco_omnidir_gamma18_robust.npz")