from keras.models import load_model as load
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

def prediction_to_char(prediction: int):
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return chars[prediction]

def load_model(path: str):
    model = load(path)
    return model

def predict(path, model):
    img = image.load_img(path, 
                   target_size=(32, 32),
                   keep_aspect_ratio = True,
                   interpolation="bilinear")
    x = image.img_to_array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    pred = model.predict(x)[0]
    return np.argmax(pred)
