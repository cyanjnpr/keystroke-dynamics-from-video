import cv2 as cv
from cv2.typing import MatLike
from typing import Deque, Tuple
from collections import deque
from numpy.typing import NDArray
from math import sqrt
from ..config import ConfigManager

status, conf = ConfigManager.read_main_config()

# TODO: it depends on font and ppi
MAX_CONTOUR_DISTANCE = 100
# TODO: it also depends on ppi
MAX_LINE_HEIGHT_DIFF = 4
# analyze frames in batch to detect anomalies, needs to be an odd number and > 4
# middle frame is current, rest is divided between previous frames and future frames
CONTOUR_BATCH_SIZE = 11

if (CONTOUR_BATCH_SIZE % 2 == 0): 
    raise ValueError("value of CONTOUR_BATCH_SIZE is not an odd number")

class CursorDetector:
    previous_frame: MatLike
    current_frame: MatLike
    contours: Deque[NDArray]

    def __init__(self, initial_frame: MatLike):
        self.current_frame = initial_frame
        self.contours = deque(maxlen=CONTOUR_BATCH_SIZE)

    def pass_frame(self, frame: MatLike) -> Tuple[bool, NDArray]:
        self.previous_frame = self.current_frame
        self.current_frame = frame
        self.detection()
        if (len(self.contours) == CONTOUR_BATCH_SIZE and not self.current_an_anomaly()):
            return True, self.contours[CONTOUR_BATCH_SIZE // 2]
        return False, None

    def contour_distance(self, c1, c2) -> float:
        x, y, _, _ = cv.boundingRect(c1)
        x_p, y_p, _, _ = cv.boundingRect(c2)
        return sqrt((x - x_p)**2 + (y - y_p)**2)

    def distance_to_last(self, contour) -> float:
        if (len(self.contours) == 0): return 0
        return self.contour_distance(contour, self.contours[-1])
        
    # Algorithm 1
    def detection(self):
        frame = cv.Canny(self.current_frame, 127, 255)
        frame_p = cv.Canny(self.previous_frame, 127, 255)
        frame_xored = cv.bitwise_xor(frame, frame_p)
        frame_contours, _ = cv.findContours(frame_xored, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        best_contour = None
        for contour in frame_contours:
            _, _, w, h = cv.boundingRect(contour)
            if (h > (conf.get_font_height() - MAX_LINE_HEIGHT_DIFF) 
                and h < (conf.get_font_height() + MAX_LINE_HEIGHT_DIFF)
                and w < 2*conf.get_font_height()):
                if (best_contour is None or self.distance_to_last(best_contour) >= self.distance_to_last(contour)):
                    best_contour = contour
        if (best_contour is not None): 
            self.contours.append(best_contour)
        
    def current_an_anomaly(self) -> bool:
        index = CONTOUR_BATCH_SIZE // 2
        for i, c in enumerate(self.contours):
            if (i != index and self.contour_distance(c, self.contours[index]) < MAX_CONTOUR_DISTANCE):
                return False
        return True
