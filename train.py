from os.path import dirname
import tensorflow as tf
from utils.settings import *
import utils.yaml
from utils.train_func import conv_block, dense_block
import os

# read params from params.yaml
params = utils.yaml.read_yaml("params.yaml")
IMAGE_WIDTH = params["train"]["image_width"]
IMAGE_HEIGHT = params["train"]["image_height"]
BATCH_SIZE = params["train"]["batch_size"]
TRAIN_PROCESSED_DATA_DIR = 'data/processed/train/'
TEST_PROCESSED_DATA_DIR = 'data/processed/test/'

# some tensorflow things
AUTOTUNE = tf.data.experimental.AUTOTUNE
METRICS = [tf.keras.metrics.AUC(name='auc')]

# define static variables
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]
EPOCHS = 5
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
NUM_CLASSES = len(class_names)
one_hot_label = lambda image, label: (image, tf.one_hot(label, NUM_CLASSES))
NUM_IMAGES = []

for label in class_names:
    dir_name = TRAIN_PROCESSED_DATA_DIR + label
    dir_length = len(os.listdir(dir_name))
    NUM_IMAGES.append(dir_length)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PROCESSED_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=2023,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, )

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PROCESSED_DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

train_ds.class_names = class_names
test_ds.class_names = class_names

train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])
    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])

    return block


model = tf.keras.Sequential([
    tf.keras.Input(shape=(*IMAGE_SIZE, 3)),

    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(),

    conv_block(16),
    conv_block(32),

    conv_block(64),
    tf.keras.layers.Dropout(0.2),

    conv_block(128),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    dense_block(256, 0.7),
    dense_block(64, 0.5),
    dense_block(32, 0.3),

    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=METRICS
)
try:
    model.load_weights('outs/model_lastsaved.h5')
except:
    pass

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('outs/model_checkpoint.h5', save_best_only=True, verbose=1)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    epochs=EPOCHS,
    verbose=1
)

model.save('outs/model_lastsaved.h5')
