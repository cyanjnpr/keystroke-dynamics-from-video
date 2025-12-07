from typing import List
from cv2.typing import MatLike
import cv2 as cv
from math import sqrt

# TODO: it depends on font and ppi
MAX_CONTOUR_DISTANCE = 100 

def contour_distance(contour, contour_p) -> float:
    x, y, _, _ = cv.boundingRect(contour)
    x_p, y_p, _, _ = cv.boundingRect(contour_p)
    return sqrt((x - x_p)**2 + (y - y_p)**2)

def cursor_detection(frame: MatLike, frame_p: MatLike) -> None:
    frame = cv.Canny(frame, 100, 200)
    frame_p = cv.Canny(frame_p, 100, 200)
    frame_xored = cv.bitwise_xor(frame, frame_p)
    frame_contours, _ = cv.findContours(frame_xored, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # look for the tallest slim line.
    best_contour = None
    for contour in frame_contours:
        _, _, w, h = cv.boundingRect(contour)
        _, _, w_b, h_b = cv.boundingRect(contour if best_contour is None else best_contour)
        if (float(h) / float(w) >= 5 and w <= 5):
            if (float(h) / float(w) >= float(h_b) / float(w_b)):
                best_contour = contour
    return best_contour

# if a contour has two distant neighbours it may be an anomaly
# if it's the last or the first it needs to have a close neigbhour
def clear_anomalies(contour_list: List[List[MatLike]]) -> List[List[MatLike]]:
    if (len(contour_list) <= 3): return
    while(contour_distance(contour_list[0][2], contour_list[1][2]) > MAX_CONTOUR_DISTANCE):
        if (len(contour_list) <= 3): return
        contour_list.pop(0)
    while(contour_distance(contour_list[-1][2], contour_list[-2][2]) > MAX_CONTOUR_DISTANCE):
        if (len(contour_list) <- 3): return
        contour_list.pop(-1)
    i = 1
    while(i < len(contour_list) - 2):
        if (contour_distance(contour_list[i - 1][2], contour_list[i][2]) > MAX_CONTOUR_DISTANCE and
            contour_distance(contour_list[i][2], contour_list[i + 1][2]) > MAX_CONTOUR_DISTANCE):
            contour_list.pop(i)
        else:
            i += 1
    # filter out lines with height not matching the height of a cursor
    average_height = sum([cv.boundingRect(contour[2])[3] for contour in contour_list]) / len(contour_list)
    return [contour for contour in contour_list if abs(cv.boundingRect(contour[2])[3] - average_height) < 2]
    