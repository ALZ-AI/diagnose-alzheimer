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

# define static variables
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]
EPOCHS = 100
class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
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
    subset="train",
    seed=2023,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PROCESSED_DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)


train_ds.class_names = class_names
test_ds.class_names = class_names

train_ds = train_ds.map(one_hot_label)
test_ds = test_ds.map(one_hot_label)

train_ds = train_ds.cache().prefetch()
test_ds = test_ds.cache().prefetch()

model = tf.keras.Sequential([
	tf.keras.Input(shape=(*IMAGE_SIZE, 3)),

	tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
	tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
	tf.keras.layers.MaxPool2D(),

	conv_block(32),
	conv_block(64),

	conv_block(128),
	tf.keras.layers.Dropout(0.2),

	conv_block(256),
	tf.keras.layers.Dropout(0.2),

	tf.keras.layers.Flatten(),
	dense_block(512, 0.7),
	dense_block(128, 0.5),
	dense_block(64, 0.3),
 
	tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])



model.save("outs/model.h5")