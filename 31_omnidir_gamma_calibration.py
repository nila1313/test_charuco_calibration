import cv2
import glob
import numpy as np

def preprocess_gamma(img, gamma=1.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(gray, table)


image_paths = sorted(glob.glob("all_good_charuco_frames/*.jpg"))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard((7, 7), 1.0, 0.8, dictionary)
all_board_corners = board.getChessboardCorners()

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 80
params.cornerRefinementMinAccuracy = 0.0005

detector = cv2.aruco.ArucoDetector(dictionary, params)

objpoints = []
imgpoints = []
image_size = None

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    image_size = (img.shape[1], img.shape[0])

    gray = preprocess_gamma(img, 1.8)

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

    if not (9 <= n <= 20):
        continue

    ids = charuco_ids.flatten()

    obj = all_board_corners[ids].astype(np.float64)
    imgp = charuco_corners.reshape(-1, 2).astype(np.float64)

    objpoints.append(obj.reshape(-1, 1, 3))
    imgpoints.append(imgp.reshape(-1, 1, 2))

print("Frames used:", len(objpoints))

# ---------- omnidir calibration ----------

w, h = image_size

K = np.array([
    [w / 2, 0, w / 2],
    [0, w / 2, h / 2],
    [0, 0, 1]
], dtype=np.float64)

xi = np.array([[1.0]], dtype=np.float64)
D = np.zeros((1, 4), dtype=np.float64)

flags = (
    cv2.omnidir.CALIB_USE_GUESS +
    cv2.omnidir.CALIB_FIX_SKEW
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

print("\n=== RESULT (gamma 1.8) ===")
print("RMS:", rms)
print("K:\n", K)
print("xi:\n", xi)
print("D:\n", D)
print("Valid frames:", idx.shape)

np.savez(
    "charuco_omnidir_gamma18.npz",
    camera_matrix=K,
    xi=xi,
    dist_coeffs=D
)

print("\nSaved: charuco_omnidir_gamma18.npz")