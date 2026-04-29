import cv2
import glob
import os
import numpy as np

input_folder = "all_good_charuco_frames"
output_folder = "debug_gamma_refinement"

os.makedirs(output_folder, exist_ok=True)

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
params.cornerRefinementMaxIterations = 80
params.cornerRefinementMinAccuracy = 0.0005

detector = cv2.aruco.ArucoDetector(dictionary, params)


# ---------- preprocessing ----------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gamma(img, g):
    gray = to_gray(img)
    table = np.array([
        ((i / 255.0) ** g) * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(gray, table)


def clahe(gray):
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return c.apply(gray)


def sharp(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.3, blur, -0.3, 0)


# ---------- methods ----------
methods = {
    "gamma_1.3": lambda img: gamma(img, 1.3),
    "gamma_1.6": lambda img: gamma(img, 1.6),
    "gamma_1.8": lambda img: gamma(img, 1.8),
    "gamma_2.0": lambda img: gamma(img, 2.0),

    "g1.6_clahe": lambda img: clahe(gamma(img, 1.6)),
    "g1.6_clahe_sharp": lambda img: sharp(clahe(gamma(img, 1.6))),
}

image_paths = sorted(glob.glob(f"{input_folder}/*.jpg"))
image_paths = image_paths[:80]  # test subset

summary = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    base_name = os.path.splitext(os.path.basename(path))[0]

    for name, func in methods.items():
        processed = func(img)

        marker_corners, marker_ids, _ = detector.detectMarkers(processed)

        marker_count = 0 if marker_ids is None else len(marker_ids)
        charuco_count = 0

        debug = img.copy()

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)

            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                processed,
                board
            )

            if charuco_ids is not None:
                charuco_count = len(charuco_ids)
                cv2.aruco.drawDetectedCornersCharuco(
                    debug,
                    charuco_corners,
                    charuco_ids,
                    (0, 0, 255)
                )

        cv2.putText(
            debug,
            f"{name} | markers:{marker_count} | charuco:{charuco_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        out_name = f"{base_name}_{name}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), debug)

        summary.append((name, marker_count, charuco_count))


# ---------- summary ----------
print("\nSummary:")
for name in methods.keys():
    vals = [s for s in summary if s[0] == name]

    avg_markers = np.mean([v[1] for v in vals])
    avg_charuco = np.mean([v[2] for v in vals])
    max_charuco = np.max([v[2] for v in vals])

    print(
        name,
        "| avg markers:", round(avg_markers, 2),
        "| avg charuco:", round(avg_charuco, 2),
        "| max charuco:", max_charuco
    )

print("\nSaved debug images in:", output_folder)