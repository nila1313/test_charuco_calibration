import cv2

print("OpenCV version:", cv2.__version__)

print("Has fisheye module:", hasattr(cv2, "fisheye"))
print("Has omnidir module:", hasattr(cv2, "omnidir"))

# Try accessing directly
try:
    print("fisheye module works:", cv2.fisheye is not None)
except:
    print("fisheye module NOT usable")

try:
    print("omnidir module works:", cv2.omnidir is not None)
except:
    print("omnidir module NOT usable")