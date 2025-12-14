from cv2.typing import MatLike
from dataclasses import dataclass
from typing import Self

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
    
    def __repr__(self):
        return f'KUnit from frame {self.frame_no}. x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}'