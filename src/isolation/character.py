from ..util import KUnit, cbb_to_ibb
from cv2.typing import MatLike
from numpy.typing import NDArray
from typing import List, Tuple
import cv2 as cv
from math import ceil
import numpy as np

CURSOR_WIDTH_CUTOFF = 3

class CharacterExtractor:
    frame_no: int
    frame: MatLike
    contour: NDArray
    
    def __init__(self, frame_no: int, frame: MatLike, contour: NDArray):
        self.frame_no = frame_no
        self.frame = frame
        self.contour = contour

    def get_mask_coords(self, mask: MatLike):
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0: return 0,0,0,0
        return cv.boundingRect(contours[0])

    def get_mask_x(self, mask: MatLike):
        return self.get_mask_coords(mask)[0]
    
    def do_masks_overlap(self, m1: MatLike, m2: MatLike) -> bool:
        x1, _, w1, _ = self.get_mask_coords(m1)
        x2, _, w2, _ = self.get_mask_coords(m2)
        return (x1 <= x2 + w2  and  x1 + w1 >= x2)
    
    def extract(self, draw_convex: bool = False) -> Tuple[bool, KUnit]:
        masks = self.extract_all()
        return self.extract_rc(masks, draw_convex)

    def extract_all(self) -> List[MatLike]:
        x, y, w, h = cbb_to_ibb(*cv.boundingRect(self.contour))
        subframe = cv.cvtColor(self.frame[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(subframe, 127, 255, 
            cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        
        num_labels, labels = cv.connectedComponents(threshold, connectivity = 4)
        masks = []
        for i in range(0, num_labels):
            mask: MatLike = (labels == i).astype("uint8") * 255
            masks.append(mask)
        # sometimes opencv detects background as the one and only component, prevent that
        masks = filter(lambda mask: np.count_nonzero(mask == 255) < mask.size / 2, masks)
        masks = sorted(masks, key = lambda mask: self.get_mask_x(mask))
        # characters may consist of mutliple components e.g. 'i'
        return self.merge_by_x(masks)
    
    def merge_by_x(self, masks: List[MatLike]) -> MatLike:
        if len(masks) < 2: return masks
        result = [masks[0]]
        for m in masks[1:]:
            if self.do_masks_overlap(result[-1], m):
                result[-1] = cv.bitwise_or(m, result[-1])
            else:
                result.append(m)
        return result
    
    def extract_rc(self, masks: List[MatLike], draw_convex: bool = False) -> Tuple[bool, KUnit]:
        if len(masks) == 0: return False, None
        mask = masks[-1]
        x, y, w, h = self.get_mask_coords(mask)
        # mask is a cursor, skip
        if (w < CURSOR_WIDTH_CUTOFF and len(masks) > 1):
            mask = masks[-2]
            x, y, w, h = self.get_mask_coords(mask)
        ibb_x, ibb_y, _, _ = cbb_to_ibb(*cv.boundingRect(self.contour))
        x += ibb_x
        y += ibb_y
        mask = self.sanitize(mask)
        if (draw_convex):
            mask = self.draw_convexity(mask)
        else:
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        return True, KUnit(self.frame_no, mask, x, y, w, h)
    
    def draw_convexity(self, mask: MatLike) -> MatLike:
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        if len(contours) == 0: return
        cnt = contours[0]
        hull = cv.convexHull(cnt, returnPoints = False)
        defects = cv.convexityDefects(cnt, hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,_ = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv.line(mask,start,end,[0,255,0],2)
                cv.circle(mask,far,2,[0,0,255],3)
        return mask

    def sanitize(self, mask: MatLike) -> MatLike:
        x, y, w, h = self.get_mask_coords(mask)
        side = max(w, h)
        mask = cv.copyMakeBorder(mask[y:y+h, x:x+w], 
            (side - h) // 2 + ceil(side / 8.), (side - h) // 2 + ceil(side / 8.),
            (side - w) // 2 + ceil(side / 8.), (side - w) // 2 + ceil(side / 8.),
            cv.BORDER_CONSTANT, value = (0, 0, 0))
        return cv.resize(mask, (128, 128), interpolation=cv.INTER_AREA)

