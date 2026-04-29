import cv2
import glob
import numpy as np

def preprocess_for_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)
    return sharp


image_paths = sorted(glob.glob("omnidir_diverse_frames/*.jpg"))

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

detector = cv2.aruco.ArucoDetector(dictionary, params)

all_board_corners = board.getChessboardCorners()

objpoints = []
imgpoints = []
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

    # Same useful range as best previous experiment
    if not (9 <= n <= 20):
        continue

    ids = charuco_ids.flatten()

    obj = all_board_corners[ids].astype(np.float64)
    imgp = charuco_corners.reshape(-1, 2).astype(np.float64)

    objpoints.append(obj.reshape(-1, 1, 3))
    imgpoints.append(imgp.reshape(-1, 1, 2))

print("Image size:", image_size)
print("Frames used:", len(objpoints))

if len(objpoints) < 10:
    print("Not enough frames for omnidir calibration.")
    exit()

w, h = image_size

# Initial guess
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

try:
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

    print("\nOmnidirectional calibration RMS error:", rms)
    print("K:\n", K)
    print("xi:\n", xi)
    print("D:\n", D)
    print("Valid frame indices:", idx.shape if idx is not None else None)

    np.savez(
        "charuco_omnidir_calibration_result.npz",
        camera_matrix=K,
        xi=xi,
        dist_coeffs=D
    )

    print("\nSaved charuco_omnidir_calibration_result.npz")

except cv2.error as e:
    print("\nOmnidirectional calibration failed:")
    print(e)