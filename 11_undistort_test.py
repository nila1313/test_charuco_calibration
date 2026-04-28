import cv2
import numpy as np
import glob
import os

os.makedirs("undistorted_test", exist_ok=True)

K = np.array([
    [783.37135688, 0, 990.47486859],
    [0, 788.64353441, 537.49990217],
    [0, 0, 1]
], dtype=np.float32)

D = np.array([-0.26075098, 0.06295615, 0.00118881, 0.00156155, -0.00651904], dtype=np.float32)

image_paths = sorted(glob.glob("good_frames/*.jpg"))

for path in image_paths[:20]:
    img = cv2.imread(path)
    h, w = img.shape[:2]

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, D, None, new_K)

    combined = np.hstack((img, undistorted))

    name = os.path.basename(path)
    cv2.imwrite(f"undistorted_test/{name}", combined)

print("Saved before/after images in undistorted_test")