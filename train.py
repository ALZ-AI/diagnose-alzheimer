import tensorflow as tf
from utils.settings import *
import utils.yaml

params = utils.yaml.read_yaml("params.yaml")

image_width = params["image_width"]
image_height = params["image_height"]
batch_size = params["batch_size"]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  ALZHEIMER_PROCESSED_DATA_DIR,
  validation_split=0.2,
  subset="train",
  seed=123,
  image_size=(image_height, image_width),
  batch_size=batch_size)

