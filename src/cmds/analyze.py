import cv2 as cv

import os
from ..resnet import load_model
from ..isolation import CursorDetector

def get_model():
    model_list = [model for model in os.listdir("models") if model.endswith(".keras")]
    model_list.sort()
    if len(model_list) == 0:
        print("No models detected. Train a model first.")
        return
    return load_model(os.path.join("models", model_list[-1]))

def analyze_command(filename):
    src = cv.VideoCapture(filename)
    s, frame = src.read()
    if not s: return

    detector = CursorDetector(frame)
    s, frame = src.read()
    while s:
        ds, c = detector.pass_frame(frame)
        if (ds):
            sh = frame.copy()
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(sh, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv.imshow("frame", sh)
            cv.waitKey(10)
            print(x, y, w ,h)
        s, frame = src.read()
    