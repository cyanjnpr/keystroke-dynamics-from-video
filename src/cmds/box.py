import cv2 as cv

from ..isolation import CursorDetector
from ..config import ConfigManager
from ..util import cbb_to_ibb, save_location
from rich.progress import Progress
from pathlib import Path
import time
from enum import Enum

status, conf = ConfigManager.read_main_config()

MARGIN_Y = int(conf.get_font_height() / 2)
MARGIN_X = int(conf.get_font_height() * 3)

class BoundingBox(Enum):
    CURSOR = 1
    ISOLATION = 2

def bb_command(kind: BoundingBox, filename: str, dest: str):
    dest_path = save_location(dest, "cbb" if kind is BoundingBox.CURSOR else "ibb")
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
                fw = min(conf.get_font_height() + 2*MARGIN_X, frame_width-x)
                fh = min(conf.get_font_height() + 2*MARGIN_Y, frame_height-y)
                if kind is BoundingBox.ISOLATION:
                    ix, iy, iw, ih = cbb_to_ibb(x, y, w, h)
                    cv.rectangle(f, (ix, iy), (ix+iw, iy+ih), (255, 0, 0), 2)
                cv.rectangle(f, (x, y), (x+w, y+h), (0, 0, 255), 2)
                f = f[fy:fy+fh, fx:fx+fw]
                cv.imwrite(str(dest_path / f"{i}.png"), f)
            s, frame = src.read()

def ibb_command(filename: str, dest: str):
    bb_command(BoundingBox.ISOLATION, filename, dest)

def cbb_command(filename: str, dest: str):
    bb_command(BoundingBox.CURSOR, filename, dest)
