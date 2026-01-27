from keras.models import load_model as load
from keras.preprocessing import image
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import string
import os
import click
from pathlib import Path

@dataclass()
class Prediction:
    character: str
    accuracy: float


def index_to_char(prediction: int):
    chars = string.digits + string.ascii_uppercase + string.ascii_lowercase
    return chars[prediction]

def load_model(source: str) -> Tuple[bool, any]:
    path = Path(source)
    if not path.exists():
        return False, None
    if path.is_dir():
        model_list = [model for model in os.listdir(str(path)) if model.endswith(".keras")]
        if len(model_list) == 0: return False, None
        model_list.sort()
        path /= model_list[-1]
    elif not str(path).endswith(".keras"): return False, None
    return load(str(path))

def predict(path, model) -> Prediction:
    img = image.load_img(path, 
                   target_size=(32, 32),
                   keep_aspect_ratio = True,
                   interpolation="bilinear")
    x = image.img_to_array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose = None)[0]

    argmax = np.argmax(pred)
    return Prediction(
        index_to_char(argmax),
        pred[argmax]
    )
