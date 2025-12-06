import cv2
import numpy as np
from cv2.typing import MatLike
from typing import Tuple

def extract_frame(vid: cv2.VideoCapture) -> Tuple[bool, MatLike]:
    status, frame = vid.read()
    if not status:
        return status, None
    return True, cv2.Canny(frame, 100, 200)

def extract_contours(frame: MatLike, frame_p: MatLike):
    xored = cv2.bitwise_xor(frame, frame_p)
    contours, hierarchy = cv2.findContours(xored, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contour_distance(contour, contour_p) -> int:
    if contour_p == None: return 0
    x, y, w, h = cv2.boundingRect(contour)
    x_p, y_p, w_p, h_p = cv2.boundingRect(contour)


def main():
    src = cv2.VideoCapture("res/video.mp4")
    frame_width = int(src.get(3))
    frame_height = int(src.get(4))
    fps = src.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("res/tracking.mp4", codec, fps, (frame_width, frame_height))

    i = 1
    contour_p = None
    status, frame_p = extract_frame(src)
    if not status: raise Exception()

    while status:
        i += 1
        status, frame = extract_frame(src)
        raw = frame
        if status:
            out_frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
            contours = extract_contours(frame, frame_p)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if (h / w >= 5 and w <= 5):
                    contour_p = contour
                    print("Frame: {}, X: {}, Y: {}, W: {}, H: {}".format(i, x, y ,w ,h))
                    # cv2.imshow('image', raw2)
                    # cv2.waitKey(50)
                    break
            frame_p = raw
            x, y, w, h = cv2.boundingRect(contour_p)
            cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0,0,255), 3)
            out.write(out_frame)

    out.release()
    src.release()
        
if __name__ == "__main__":
    main()