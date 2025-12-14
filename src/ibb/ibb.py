import cv2 as cv
from cv2.typing import MatLike
from typing import Tuple
from math import ceil
import config
from .kunit import KUnit

CURSOR_WIDTH_CUTOFF = 3

status, conf = config.ConfigManager.read_main_config()

# isolation bounding box
def cbb_to_ibb(x, y, w, h) -> Tuple[int, int, int, int]:
    return (x - h * 2, y, w + h * 2, h)

# rightmost character
def extract_rc(frame_no: int, frame: MatLike, contour: MatLike) -> KUnit:
    x_cbb, y_cbb, w_cbb, h_cbb = cv.boundingRect(contour)
    x, y, w, h = cbb_to_ibb(x_cbb, y_cbb, w_cbb, h_cbb)
    subframe = cv.cvtColor(frame[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)

    _, threshold = cv.threshold(subframe, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    num_labels, labels = cv.connectedComponents(threshold, connectivity = 4)

    # we expect at least cursor and rightmost character
    if (num_labels < 2): return None

    masks = []
    for i in range(0, num_labels):
        mask: MatLike = (labels == i).astype("uint8") * 255
        masks.append(mask)

    def get_mask_key(mask):
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0: return cv.boundingRect(contours[0])[0]
        return 0

    masks = sorted(masks, key = lambda mask: get_mask_key(mask))
    # [-1] is often a cursor (not always, figure 5 and 6 in the paper)
    rc = prepare_rc(masks[-1])
    rc_x, rc_y, rc_w, rc_h = rc_coords(masks[-1])
    # if it is a cursor, next edge has(?) to be a rc
    if (rc_w < CURSOR_WIDTH_CUTOFF):
        rc = prepare_rc(masks[-2])
        rc_x, rc_y, rc_w, rc_h = rc_coords(masks[-2])
    rc_x += x
    rc_y += y
    return KUnit(frame_no, rc, rc_x, rc_y, rc_w, rc_h)

def rc_coords(rc: MatLike) -> Tuple[int, int, int, int]:
    contours, _ = cv.findContours(rc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return 0,0,0,0
    return cv.boundingRect(contours[0])

# center character on a square with even margins
def prepare_rc(rc: MatLike) -> MatLike:
    contours, _ = cv.findContours(rc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return rc
    x, y, w, h = cv.boundingRect(contours[0])
    side = max(w, h)

    rc = cv.copyMakeBorder(rc[y:y+h, x:x+w], 
                      (side - h) // 2 + ceil(side / 8.), (side - h) // 2 + ceil(side / 8.),
                      (side - w) // 2 + ceil(side / 8.), (side - w) // 2 + ceil(side / 8.),
                      cv.BORDER_CONSTANT,
                      value = (0, 0, 0))
    
    # reverse mask
    _, rc = cv.threshold(rc, 127, 255, cv.THRESH_BINARY_INV)
    return rc
