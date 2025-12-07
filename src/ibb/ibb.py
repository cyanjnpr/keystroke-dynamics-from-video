import cv2 as cv
from cv2.typing import MatLike
from typing import Tuple

# isolation bounding box
def cbb_to_ibb(x, y, w, h) -> Tuple[int, int, int, int]:
    return (x - h, y, w + h, h)

# rightmost character
def extract_rc(time: float, frame: MatLike, contour: MatLike) -> MatLike:
    x_cbb, y_cbb, w_cbb, h_cbb = cv.boundingRect(contour)
    x, y, w, h = cbb_to_ibb(x_cbb, y_cbb, w_cbb, h_cbb)
    subframe = cv.cvtColor(frame[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)

    _, threshold = cv.threshold(subframe, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    num_labels, labels = cv.connectedComponents(threshold, connectivity = 4)

    # we expect at least cursor and rightmost character
    if (num_labels < 2): return None

    masks = []
    for i in range(0, num_labels):
        mask = (labels == i).astype("uint8") * 255
        masks.append(mask)
    masks = sorted(masks, key = lambda mask: cv.boundingRect(mask)[0])
    for i, mask in enumerate(masks):
        cv.imwrite("res/{}.png".format(i), mask)
    # [-1] is cursor (not always, figure 5 and 6 in the paper)
    return masks[-2]