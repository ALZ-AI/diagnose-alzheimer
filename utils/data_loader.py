import tensorflow as tf
from utils.settings import *


def load_data(image_size, batch_size, class_names, autotune):
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PROCESSED_DATA_DIR,
    color_mode="grayscale",
    image_size=image_size,
    batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_PROCESSED_DATA_DIR,
        image_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size)

    train_ds.class_names = class_names
    test_ds.class_names = class_names
    
    NUM_CLASSES = len(class_names)
    one_hot_label = lambda image, label: (image, tf.one_hot(label, NUM_CLASSES))

    train_ds = train_ds.map(one_hot_label, num_parallel_calls=autotune)
    test_ds = test_ds.map(one_hot_label, num_parallel_calls=autotune)

    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)
    
    return train_ds, test_ds