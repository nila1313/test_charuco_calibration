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


image_paths = sorted(glob.glob("all_good_charuco_frames/*.jpg"))

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

objpoints = []
imgpoints = []
image_size = None

all_board_corners = board.getChessboardCorners()

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

    if not (9 <= n <= 20):
        continue

    ids = charuco_ids.flatten()

    obj = all_board_corners[ids].astype(np.float32)
    imgp = charuco_corners.reshape(-1, 2).astype(np.float32)

    objpoints.append(obj.reshape(-1, 1, 3))
    imgpoints.append(imgp.reshape(-1, 1, 2))

print("Image size:", image_size)
print("Frames used:", len(objpoints))

if len(objpoints) < 10:
    print("Not enough frames for fisheye calibration.")
    exit()

K = np.zeros((3, 3))
D = np.zeros((4, 1))

flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_CHECK_COND
    + cv2.fisheye.CALIB_FIX_SKEW
)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-6
)

try:
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        None,
        None,
        flags,
        criteria
    )

    print("\nFisheye calibration RMS error:", rms)
    print("K:\n", K)
    print("D:\n", D)

    np.savez(
        "charuco_fisheye_calibration_result.npz",
        camera_matrix=K,
        dist_coeffs=D
    )

    print("\nSaved charuco_fisheye_calibration_result.npz")

except cv2.error as e:
    print("\nFisheye calibration failed:")
    print(e)