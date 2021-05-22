import tensorflow as tf
import numpy as np
from PIL import Image
import os
import PIL
import glob

from keras.preprocessing import image
img = image.load_img("normal.png", target_size=(256, 256))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

model = tf.keras.models.load_model("../outs/model_lastsaved.h5")

prediction= model.predict(img)
print(prediction)
