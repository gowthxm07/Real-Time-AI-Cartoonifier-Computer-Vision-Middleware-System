import cv2
import numpy as np

print(f"OpenCV Version: {cv2.__version__}")

try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    if count > 0:
        print(f"Success! {count} CUDA device(s) detected.")
    else:
        print("No CUDA devices detected. (This is expected for Iris Xe)")
except AttributeError:
    print("CUDA module not found in this OpenCV build. System will default to CPU mode.")