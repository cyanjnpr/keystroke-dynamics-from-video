import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
import keras

# Paremeters used here are obtained from
# the Table 2 'ResNet hyperparameters' from the paper
def get_datasets(dataset_dir: str):
    train_ds = keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split = 0.2,
        subset = "training",
        label_mode = 'categorical',
        seed = 1337,
        image_size = (32, 32),
        batch_size = 64
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split = 0.2,
        subset = "validation",
        label_mode = 'categorical',
        seed = 1337,
        image_size = (32, 32),
        batch_size = 64
    )
    return train_ds, val_ds


def train_model(dataset_dir: str, num_classes: int):
    train_ds, val_ds = get_datasets(dataset_dir)

    # rescale pixel values from [0,255] to [0,1]
    # (section under the Table 2)
    normalization_layer = keras.layers.Rescaling(1. / 255)
    norm_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    norm_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    base_model = ResNet50(
        weights = "imagenet",
        include_top = False,
        input_shape = (32, 32, 3)
        )
    
    base_model.layers.pop()
    for layer in base_model.layers:
        layer.trainable = False

    inputs = keras.Input(shape = (32, 32, 3))
    x = base_model(inputs, training = False)
    # flatten the tensor or whatever
    # (1, 1, num_classes) -> (num_classes)
    x = GlobalAveragePooling2D()(x)

    x = Dense(num_classes, activation="softmax")(x)

    model = Model(
        inputs = inputs, 
        outputs = x
    )

    # still in Table 2
    model.compile(
        optimizer = SGD(learning_rate = 0.1),
        loss='categorical_crossentropy', 
        metrics=['accuracy'] 
    )

    start = time.time()
    history = model.fit(
        norm_train_ds,
        validation_data = norm_val_ds,
        batch_size = 64,
        epochs = 40,
    )
    stop = time.time()
    logging.info(f'Training took: {(stop - start) / 60} minutes')

    return model, history, train_ds.class_names


def do_train(dataset_dir: str = "dataset", models_dir: str = "models"):
    # set up logging
    save_to = os.path.join(models_dir, time.strftime("%Y%m%d%H%M"))
    try:
        os.mkdir(save_to)
    except OSError:
        print("Creation of the directory %s failed" % save_to)

    logging.basicConfig(level = logging.DEBUG,
                        filename=os.path.join(save_to, 'training.log'))
                        # stream=sys.stdout)
    logging.debug(f'training folder: {dataset_dir}')

    num_classes = len([f.path for f in os.scandir(dataset_dir) if f.is_dir()])
    model, history, class_names = train_model(dataset_dir,  num_classes)
    model.save(os.path.join(save_to, "model.keras"))

    return model, history
