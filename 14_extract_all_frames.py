import cv2
import os

video_path = "calibration_video.MP4"   # change name if needed
out_dir = "all_video_frames"

os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # save every 2nd frame
    if frame_id % 2 == 0:
        cv2.imwrite(f"{out_dir}/frame_{frame_id:05d}.jpg", frame)
        saved += 1

    frame_id += 1

cap.release()

print("Total video frames:", frame_id)
print("Saved frames:", saved)