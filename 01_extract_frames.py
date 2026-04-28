import cv2
import os

video_path = "calibration_video.mp4"
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % 10 == 0:
        cv2.imwrite(f"{out_dir}/frame_{saved:04d}.jpg", frame)
        saved += 1

    frame_id += 1

cap.release()
print("Saved frames:", saved)