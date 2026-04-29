import cv2
import glob
import os
import numpy as np

input_folder = "all_good_charuco_frames"
output_folder = "debug_preprocessing_compare"

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


def gray_original(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def clahe_only(img):
    gray = gray_original(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def clahe_sharp(img):
    gray = clahe_only(img)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.3, blur, -0.3, 0)


def gamma_dark(img):
    gray = gray_original(img)
    gamma = 1.6
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(gray, table)


def adaptive_equalized(img):
    gray = gray_original(img)
    gray = cv2.equalizeHist(gray)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )


methods = {
    "original": gray_original,
    "clahe": clahe_only,
    "clahe_sharp": clahe_sharp,
    "gamma_dark": gamma_dark,
    "adaptive_equalized": adaptive_equalized,
}

image_paths = sorted(glob.glob(f"{input_folder}/*.jpg"))

# test first 80 images for now
image_paths = image_paths[:80]

summary = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    base_name = os.path.splitext(os.path.basename(path))[0]

    for method_name, method_func in methods.items():
        processed = method_func(img)

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
            f"{method_name} | markers: {marker_count} | charuco: {charuco_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        out_name = f"{base_name}_{method_name}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), debug)

        summary.append((method_name, marker_count, charuco_count))


print("\nSummary by method:")
for method_name in methods.keys():
    vals = [s for s in summary if s[0] == method_name]
    avg_markers = np.mean([v[1] for v in vals])
    avg_charuco = np.mean([v[2] for v in vals])
    max_charuco = np.max([v[2] for v in vals])

    print(
        method_name,
        "| avg markers:", round(avg_markers, 2),
        "| avg charuco:", round(avg_charuco, 2),
        "| max charuco:", max_charuco
    )

print("\nSaved debug images in:", output_folder)