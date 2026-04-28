import cv2
import glob

image_paths = sorted(glob.glob("selected_frames/*.jpg"))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board_sizes = [
    (5, 5), (6, 5), (7, 5), (8, 5), (9, 5),
    (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
    (5, 7), (6, 7), (7, 7), (8, 7), (9, 7),
    (10, 7),
]

marker_lengths = [0.5, 0.6, 0.7, 0.75, 0.8]

results = []

for board_size in board_sizes:
    for marker_len in marker_lengths:
        board = cv2.aruco.CharucoBoard(
            board_size,
            1.0,
            marker_len,
            dictionary
        )

        detector = cv2.aruco.CharucoDetector(board)

        good = 0
        total = 0

        for path in image_paths:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

            if charuco_ids is not None and len(charuco_ids) >= 4:
                good += 1
                total += len(charuco_ids)

        results.append((good, total, board_size, marker_len))

results = sorted(results, reverse=True)

print("Best ChArUco settings:")
for r in results[:20]:
    print(r)