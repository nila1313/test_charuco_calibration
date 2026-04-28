import cv2
import glob
import lensboy as lb

image_paths = sorted(glob.glob("good_frames/*.jpg"))
images = [cv2.imread(p) for p in image_paths]
images = [img for img in images if img is not None]

if len(images) == 0:
    raise RuntimeError("No images found in good_frames/")

print("Images loaded:", len(images))

h, w = images[0].shape[:2]

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard(
    (7, 7),
    1.0,
    0.8,
    dictionary
)

target_points, frames, image_indices = lb.extract_frames_from_charuco(
    board,
    images
)

print("Detected frames:", len(frames))

config = lb.OpenCVConfig(
    image_height=h,
    image_width=w,
)

result = lb.calibrate_camera(
    target_points,
    frames,
    camera_model_config=config,
)

print(result)

result.camera_model.save("lensboy_camera.json")
print("Saved lensboy_camera.json")