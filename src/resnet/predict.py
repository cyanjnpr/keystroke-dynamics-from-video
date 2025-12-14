from keras.models import load_model as load
from keras.preprocessing import image
import numpy as np
from dataclasses import dataclass

@dataclass()
class Prediction:
    character: str
    accuracy: float


def index_to_char(prediction: int):
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return chars[prediction]

def load_model(path: str):
    model = load(path)
    return model

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
