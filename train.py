import tensorflow as tf
from utils.settings import *
import utils.yaml
from utils.train_func import conv_block, dense_block, exponential_decay
import json
from matplotlib import pyplot as plt
# read params from params.yaml
params = utils.yaml.read_yaml("params.yaml")
IMAGE_WIDTH = params["train"]["image_width"]
IMAGE_HEIGHT = params["train"]["image_height"]
BATCH_SIZE = params["train"]["batch_size"]
LEARNING_RATE = params["train"]["learning_rate"]
EPOCHS = params["train"]["epochs"]

# some tensorflow things
AUTOTUNE = tf.data.experimental.AUTOTUNE
METRICS = ["accuracy", tf.keras.metrics.AUC(curve='ROC'), "mae"]

# define static variables
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
NUM_CLASSES = len(class_names)
one_hot_label = lambda image, label: (image, tf.one_hot(label, NUM_CLASSES))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PROCESSED_DATA_DIR,
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PROCESSED_DATA_DIR,
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE)

train_ds.class_names = class_names
test_ds.class_names = class_names

train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(*IMAGE_SIZE, 1)),
    
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

model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=METRICS
)

try:
    model.load_weights('outs/model_lastsaved.h5')
except:
    pass

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

metrics = json.dumps({
    "auc": history.history["auc"],
    "accuracy": history.history["accuracy"],
    "mae": history.history["mae"]
})

metrics_file = open("outs/metrics.json", "w+")

metrics_file.write(metrics)

metrics_file.close()

model.save('outs/model.h5')


fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['auc', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
