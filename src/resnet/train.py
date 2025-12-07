import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
import keras

# Paremeters used here are obtained from
# the Table 2 'ResNet hyperparameters' from the paper
def get_datasets(dataset_dir: str):

    train_ds = keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split = 0.2,
        subset = "training",
        label_mode = 'int',
        # color_mode = 'grayscale',
        seed = 1337,
        image_size = (32, 32),
        batch_size = 64
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split = 0.2,
        subset = "validation",
        label_mode = 'int',
        # color_mode = 'grayscale',
        seed = 1337,
        image_size = (32, 32),
        batch_size = 64
    )

    return train_ds, val_ds


def train_model(dataset_dir: str, num_classes: int):
    # split dataset into training and validation set
    train_ds, val_ds = get_datasets(dataset_dir)

    # normalize images (rescale pixel values from [0,255] to [0,1])
    # (section under the Table 2)
    normalization_layer = keras.layers.Rescaling(1. / 255)
    norm_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    norm_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # w, h, channels
    input_shape = (32, 32, 3)
    base_model = ResNet50(
        include_top = False,
        )
    
    base_model.layers.pop()
    for layer in base_model.layers:
        layer.trainable = False
    last = base_model.layers[-1].output
    x = Dense(num_classes, activation="softmax")(last)

    model = Model(
        inputs = base_model.input, 
        outputs = x
    )

    # still in Table 2
    model.compile(
        optimizer = SGD(learning_rate = 0.1),
        loss='categorical_crossentropy', 
        metrics=['accuracy'] 
    )

    # configure early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",  # monitor validation loss (that is, the loss computed for the validation holdout)
            min_delta=1e-2,  # "no longer improving" being defined as "an improvement lower than 1e-2"
            patience=10,  # "no longer improving" being further defined as "for at least 10 consecutive epochs"
            verbose=1
        )
    ]

    # train model
    start = time.time()
    # with tf.device(device_name):
    history = model.fit(
        norm_train_ds,
        validation_data = norm_val_ds,
        epochs = 40,
        callbacks=callbacks,
    )
    stop = time.time()
    logging.info(f'Training took: {(stop - start) / 60} minutes')

    return model, history, train_ds.class_names


def do_train(dataset_dir: str = "src/resnet/dataset/mono"):

    # set up logging
    save_to = os.path.join("src/resnet/models", time.strftime("%Y%m%d%H%M"))
    try:
        os.mkdir(save_to)
    except OSError:
        print("Creation of the directory %s failed" % save_to)

    logging.basicConfig(level = logging.DEBUG,
                        filename=os.path.join(save_to, 'training.log'))
                        # stream=sys.stdout)

    logging.debug(f'training folder: {dataset_dir}')

    # number of classes in training dataset
    num_classes = len([f.path for f in os.scandir(dataset_dir) if f.is_dir()])

    # train model
    model, history, class_names = train_model(dataset_dir,  num_classes)

    # save trained model
    model.save(save_to)

    return model, history
