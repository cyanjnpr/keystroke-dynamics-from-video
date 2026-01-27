import cv2 as cv
from rich.progress import Progress
import tempfile
import os

from ..resnet import load_model, predict
from ..isolation import CursorDetector, CharacterExtractor
from ..util import save_location

def kunit_command(filename: str, dest: str, convexity: bool, predictions: bool):
    dest_path = save_location(dest, "kunit")
    src = cv.VideoCapture(filename)
    frame_total = int(src.get(cv.CAP_PROP_FRAME_COUNT))
    s, frame = src.read()
    if not s: return

    if (predictions):
        model = load_model()
        fd, filename = tempfile.mkstemp(suffix=".png")
        os.close(fd)

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
                    if (predictions):
                        cv.imwrite(filename, rc.get_image())
                        p = predict(filename, model)
                        cv.imwrite(str(dest_path / f"{i}.png"), rc.image_repr_info(
                            f"Predicted Character: {p.character}",
                            f"Confidence: {p.accuracy*100:.2f}%"
                        ))
                    else:
                        cv.imwrite(str(dest_path / f"{i}.png"), rc.image_repr())
            s, frame = src.read()
        