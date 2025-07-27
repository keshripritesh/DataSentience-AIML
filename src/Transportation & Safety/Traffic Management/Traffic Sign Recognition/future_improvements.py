import cv2
from utils import preprocess_image

def process_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        processed = preprocess_image(frame)  # Pass frame directly