from utils.settings import *
from PIL import Image, ImageFile
import numpy as np
import tensorflow
import glob
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

evaluation_metrics = {}

for label in classes:
    evaluation_metrics[label] = {
        "length": 0, "TP": 0
    }
    for image_path in glob.glob(os.path.join(ALZHEIMER_PROCESSED_DATA_DIR, "test", label, "*.jpg")):
        
        image = Image.open(image_path).convert("L")

        image_np = np.array(image)
        image_np = np.array([np.array([image_np])]).reshape((256, 256, 1, 1))

        model = tensorflow.keras.models.load_model("outs/model.h5")

        prediction_matrix = model.predict(np.array([image_np]))
        prediction_matrix = prediction_matrix[0]
        
        # get results
        index = int(np.argmax(prediction_matrix))
        prediction = classes[index]
        #print("Label:", label, "Prediction:", prediction)
        evaluation_metrics[label]["length"] += 1
        if classes[index] == prediction:
            evaluation_metrics[label]["TP"] += 1
    
    evaluation_metrics[label]["accuracy"] = evaluation_metrics[label]["TP"] / evaluation_metrics["length"]


evaluation_metrics_file = open("outs/evaluation_metrics.json", "w+")

evaluation_metrics_file.write(json.dumps(evaluation_metrics))

evaluation_metrics_file.close()
