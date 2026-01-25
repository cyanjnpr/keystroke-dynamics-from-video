from cv2.typing import MatLike
from dataclasses import dataclass
from typing import Self
import cv2 as cv

@dataclass()
class KUnit:
    frame_no: int
    image: MatLike
    x: int
    y: int
    w: int
    h: int

    def is_the_same(self, that: Self) -> bool:
        w = int((self.w + that.w) / 2.)
        h = int((self.h + that.h) / 2.)
        return (abs(self.x - that.x) <= w and abs(self.y - that.y) <= h)
    
    def image_repr(self) -> MatLike:
        result = cv.bitwise_not(self.image)
        result = cv.copyMakeBorder(result, 
            64, 64, 64, 64,
            cv.BORDER_CONSTANT, value = (0, 0, 0))
        info = [
            f"Frame: {self.frame_no}",
            f"xmin: {self.x}",
            f"xmax: {self.x + self.w}"]
        dy = 0
        for line in info:
            size, ldy = cv.getTextSize(line, cv.FONT_HERSHEY_PLAIN, 1, 1)
            cv.putText(result, line, (10, 10+size[1]+dy), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            dy += size[1] + ldy
        return result
    
    def __repr__(self) -> str:
        return f'KUnit from frame {self.frame_no}. x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}'
    