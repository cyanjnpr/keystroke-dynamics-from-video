from keras.models import load_model as load
from keras.preprocessing import image
import numpy as np
from dataclasses import dataclass
import string
import os
import click

@dataclass()
class Prediction:
    character: str
    accuracy: float


def index_to_char(prediction: int):
    chars = string.digits + string.ascii_uppercase + string.ascii_lowercase
    return chars[prediction]

def load_model(source: str = "models"):
    model_list = [model for model in os.listdir("models") if model.endswith(".keras")]
    model_list.sort()
    if len(model_list) == 0:
        click.echo("No models detected. Train a model first.")
        return
    return load(os.path.join("models", model_list[-1]))

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
