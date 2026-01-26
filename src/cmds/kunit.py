import cv2 as cv
from rich.progress import Progress
from pathlib import Path
import time

from ..isolation import CursorDetector, CharacterExtractor
from ..util import save_location

def kunit_command(filename: str, dest: str, convexity: bool):
    dest_path = save_location(dest, "kunit")
    src = cv.VideoCapture(filename)
    frame_total = int(src.get(cv.CAP_PROP_FRAME_COUNT))
    s, frame = src.read()
    if not s: return

    detector = CursorDetector(frame)
    s, frame = src.read()
    i = -1
    with Progress() as progress:
        task = progress.add_task(f"Detecting Rightmost Characters...", total=frame_total)
        while s:
            i += 1
            progress.update(task, advance=1)
            ds, c = detector.pass_frame(frame)
            if (ds):
                extractor = CharacterExtractor(i, frame.copy(), c)
                es, rc = extractor.extract(convexity)
                if (es):
                    cv.imwrite(str(dest_path / f"{i}.png"), rc.image_repr())
            s, frame = src.read()
        