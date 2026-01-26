import cv2 as cv

import tempfile
import os
import csv
from rich.progress import Progress
from typing import List
from ..resnet import load_model, predict
from ..isolation import CursorDetector, CharacterExtractor
from ..util import KeyStrokePoint, save_location

def get_model():
    model_list = [model for model in os.listdir("models") if model.endswith(".keras")]
    model_list.sort()
    if len(model_list) == 0:
        print("No models detected. Train a model first.")
        return
    return load_model(os.path.join("models", model_list[-1]))

def analyze_command(filename: str, dest: str):
    dest_path = save_location(dest, "dynamics")
    src = cv.VideoCapture(filename)
    frame_total = int(src.get(cv.CAP_PROP_FRAME_COUNT))
    fps = float(src.get(cv.CAP_PROP_FPS))
    s, frame = src.read()
    if not s: return

    keystrokes: List[KeyStrokePoint] = [KeyStrokePoint()]
    model = get_model()
    fd, filename = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    detector = CursorDetector(frame)
    s, frame = src.read()
    i = -1
    with Progress() as progress:
        task = progress.add_task(f"Analyzing...", total=frame_total)
        while s:
            i += 1
            progress.update(task, advance=1)
            ds, c = detector.pass_frame(frame)
            if (ds):
                extractor = CharacterExtractor(i, frame.copy(), c)
                es, rc = extractor.extract()
                if (es):
                    if (not keystrokes[-1].is_part_of(rc)):
                        keystrokes.append(KeyStrokePoint())
                    cv.imwrite(filename, rc.get_image())
                    p = predict(filename, model)
                    keystrokes[-1].add_unit(i, rc, p)
                    keystrokes[-1].calculate_delay(fps)
            s, frame = src.read()
    with open(str(dest_path / "biometry.csv"), "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["KeyPress", "KeyRelease", "KeyDelay", "KeyText", "Confidence"])
        for k in keystrokes:
            if k.confidence > .7:
                writer.writerow([k.key_press, k.key_release, k.key_delay, k.key_text, k.confidence])
