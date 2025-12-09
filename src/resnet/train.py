import os
import time
import logging

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
import keras

# Paremeters used here are obtained from
# the Table 2 'ResNet hyperparameters' from the paper
PAPER_LEARNING_RATE = 0.1
PAPER_BATCH_SIZE = 64
PAPER_EPOCH = 40
PAPER_IMAGE_SIDE = 32
PAPER_VALIDATION_SPLIT = 0.2

def get_datasets(dataset_dir: str):
    seed = time.time_ns() & 2**31
    train_ds = keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split = PAPER_VALIDATION_SPLIT,
        subset = "training",
        label_mode = 'categorical',
        seed = seed,
        image_size = (PAPER_IMAGE_SIDE, PAPER_IMAGE_SIDE),
        batch_size = PAPER_BATCH_SIZE
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split = PAPER_VALIDATION_SPLIT,
        subset = "validation",
        label_mode = 'categorical',
        seed = seed,
        image_size = (PAPER_IMAGE_SIDE, PAPER_IMAGE_SIDE),
        batch_size = PAPER_BATCH_SIZE
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
        input_shape = (PAPER_IMAGE_SIDE, PAPER_IMAGE_SIDE, 3)
        )

    inputs = keras.Input(shape = (PAPER_IMAGE_SIDE, PAPER_IMAGE_SIDE, 3))
    x = base_model(inputs, training = True)
    # (1, 1, num_classes) -> (num_classes)
    x = GlobalAveragePooling2D()(x)
    # since all layers are trainable, try to reduce overfiting
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(
        inputs = inputs, 
        outputs = x
    )

    # Table 2
    model.compile(
        optimizer = SGD(learning_rate = PAPER_LEARNING_RATE),
        loss='categorical_crossentropy', 
        metrics=['accuracy'] 
    )

    start = time.time()
    history = model.fit(
        norm_train_ds,
        validation_data = norm_val_ds,
        batch_size = PAPER_BATCH_SIZE,
        epochs = PAPER_EPOCH,
    )
    stop = time.time()
    logging.info(f'Training took: {(stop - start) / 60} minutes')

    return model, history


def train(dataset_dir: str, models_dir: str):
    timestamp = time.strftime("%Y%m%d%H%M")
    logging.basicConfig(
        level = logging.DEBUG,
        filename=os.path.join(models_dir, f'training-{timestamp}.log'))
    logging.debug(f'training folder: {dataset_dir}')

    num_classes = len([f.path for f in os.scandir(dataset_dir) if f.is_dir()])
    model, history = train_model(dataset_dir,  num_classes)
    model.save(os.path.join(models_dir, f'model-{timestamp}.keras'))
    return model, history
