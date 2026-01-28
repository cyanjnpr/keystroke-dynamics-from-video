from cv2.typing import MatLike
from dataclasses import dataclass
from typing import List
from .kunit import KUnit
from ..resnet import Prediction
import cv2 as cv

@dataclass()
class KeyStrokePoint():
    key_press: int
    key_release: int
    key_delay: float
    key_text: str
    confidence: float
    units: List[KUnit]

    def __init__(self):
        self.units = []
        self.confidence = 0
        self.key_text = " "

    def is_part_of(self, unit: KUnit) -> bool:
        if (len(self.units) == 0): return True
        return self.units[-1].is_the_same(unit)

    def add_unit(self, frame_no: int, unit: KUnit, prediction: Prediction):
        if (prediction.accuracy > self.confidence):
            self.confidence = prediction.accuracy
            self.key_text = prediction.character
        if (len(self.units) == 0): self.key_press = frame_no
        self.key_release = frame_no
        self.units.append(unit)

    def calculate_delay(self, fps: float):
        ikt =  (self.key_release - self.key_press + 1)
        self.key_delay = (1000 / fps) * ikt
