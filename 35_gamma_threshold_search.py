import cv2
import glob
import numpy as np

image_folder = "all_good_charuco_frames"
gamma_value = 1.8

def preprocess_gamma(img, gamma=1.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(gray, table)


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

image_paths = sorted(glob.glob(f"{image_folder}/*.jpg"))

def collect_points(min_corners, max_corners):
    objpoints = []
    imgpoints = []
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

    return objpoints, imgpoints, image_size


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

    return rms, K, xi, D, idx


experiments = [
    (9, 20),
    (9, 23),
    (10, 20),
    (10, 23),
    (11, 20),
    (11, 23),
    (12, 20),
    (12, 23),
]

best = None

for min_c, max_c in experiments:
    print("\n==============================")
    print(f"Testing corners {min_c}-{max_c}")
    print("==============================")

    objpoints, imgpoints, image_size = collect_points(min_c, max_c)

    print("Frames collected:", len(objpoints))

    if len(objpoints) < 20:
        print("Skipping: too few frames")
        continue

    try:
        rms, K, xi, D, idx = calibrate_omnidir(objpoints, imgpoints, image_size)

        print("RMS:", rms)
        print("K:\n", K)
        print("xi:\n", xi)
        print("D:\n", D)
        print("Valid frames:", idx.shape if idx is not None else None)

        if best is None or rms < best["rms"]:
            best = {
                "range": (min_c, max_c),
                "rms": rms,
                "K": K,
                "xi": xi,
                "D": D,
                "frames": len(objpoints),
                "idx": idx
            }

    except cv2.error as e:
        print("Calibration failed:")
        print(e)


print("\n==============================")
print("BEST RESULT")
print("==============================")

if best is not None:
    print("Corner range:", best["range"])
    print("Frames:", best["frames"])
    print("RMS:", best["rms"])
    print("K:\n", best["K"])
    print("xi:\n", best["xi"])
    print("D:\n", best["D"])

    np.savez(
        "charuco_omnidir_gamma18_best_threshold.npz",
        camera_matrix=best["K"],
        xi=best["xi"],
        dist_coeffs=best["D"]
    )

    print("\nSaved charuco_omnidir_gamma18_best_threshold.npz")
else:
    print("No successful calibration.")