
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_vgg16_unet
from metrics import dice_loss, dice_coef, iou
import keras;

# from unetmodel import build_unet

H = 512
W = 512

def create_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_train_data():
    x = sorted(glob('C:/new_data/train_png/images/*.png'))
    y = sorted(glob('C:/new_data/train_png/masks/*.png'))
    return x, y

def load_val_data():
    x = sorted(glob("C:/new_data/val_png/images/*.png"))
    y = sorted(glob("C:/new_data/val_png/masks/*.png"))
    return x, y
    

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W,3])
    y.set_shape([H, W,1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 6
    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "data.csv")

  

    train_x, train_y = load_train_data()
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_val_data()

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    
    
    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    
    model = build_vgg16_unet((H, W, 3))

    model.compile(loss=dice_loss, optimizer=Adam(lr) )

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
    # model.fit(train_dataset, epochs=5)

    model.fit(

        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        shuffle=False

    )