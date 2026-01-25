import cv2 as cv

from ..isolation import CursorDetector
from rich.progress import Progress
from pathlib import Path
import time
from enum import Enum

MARGIN_Y = 20
MARGIN_X = 100

class BoundingBox(Enum):
    CURSOR = 1
    ISOLATION = 2

    def save_location(self, dest: str) -> Path:
        timestamp = time.strftime("%Y%m%d%H%M%S")
        path = Path(dest) / f"{"cbb" if self is BoundingBox.CURSOR else "ibb"}-{timestamp}"
        path.mkdir(parents=True, exist_ok=False)
        return path

def bb_command(kind: BoundingBox, filename: str, dest: str):
    dest_path = kind.save_location(dest)
    src = cv.VideoCapture(filename)
    frame_width = int(src.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(src.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_total = int(src.get(cv.CAP_PROP_FRAME_COUNT))
    s, frame = src.read()
    if not s: return

    detector = CursorDetector(frame)
    s, frame = src.read()
    i = 0
    with Progress() as progress:
        task = progress.add_task(f"Detecting {kind.name.capitalize()} Bounding Boxes...", total=frame_total)
        while s:
            i += 1
            progress.update(task, advance=1)
            ds, c = detector.pass_frame(frame)
            if (ds):
                f = frame.copy()
                x, y, w, h = cv.boundingRect(c)
                fx = max(x - MARGIN_X, 0)
                fy = max(y - MARGIN_Y, 0)
                fw = min(w + 2*MARGIN_X, frame_width)
                fh = min(h + 2*MARGIN_Y, frame_height)
                if kind is BoundingBox.ISOLATION:
                    diff = max(2*h - w, 0)
                    cv.rectangle(f, (x-diff, y), (x+w, y+h), (255, 0, 0), 2)
                cv.rectangle(f, (x, y), (x+w, y+h), (0, 0, 255), 2)
                f = f[fy:fy+fh, fx:fx+fw]
                cv.imwrite(str(dest_path / f"{i}.png"), f)
            s, frame = src.read()

def ibb_command(filename: str, dest: str):
    bb_command(BoundingBox.ISOLATION, filename, dest)

def cbb_command(filename: str, dest: str):
    bb_command(BoundingBox.CURSOR, filename, dest)
