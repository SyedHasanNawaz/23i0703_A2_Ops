import tensorflow as tf
import pandas as pd
import numpy as np

IMG_SIZE = (224, 224)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_crop(img, (200, 200, 3))
    img = tf.image.resize(img, IMG_SIZE)
    return img, label

def make_ds(csv_path, training=False, batch_size=BATCH):
    df = pd.read_csv(csv_path)
    paths = df["path"].values
    labels = df["label"].values.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(1000, seed=42).map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def preprocess_single(image_array):
    arr = np.asarray(image_array, dtype=np.float32)
    arr = tf.image.resize(arr, IMG_SIZE).numpy()
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)