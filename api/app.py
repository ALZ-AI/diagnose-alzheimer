import json
import tensorflow as tf
import boto3
import tempfile
from cgi import parse_header, parse_multipart
import base64
import io
from PIL import Image, ImageFile
import numpy as np

BUCKET_NAME = "diagnose-alzheimer-bucket"
MODEL_FILE_PATH = 'outs/model.h5'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_PATH

ImageFile.LOAD_TRUNCATED_IMAGES = True

def predict(event, context):
    
    # check get or post
    data = json.loads(event["body"])
    decoded_string = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(decoded_string)).resize((256, 256)).convert("RGB")
    image_np = np.array(image)
    image_np = np.array([image_np])
    
    # download model from S3 Bucket to lambda
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(BUCKET_NAME)
    object = bucket.Object(MODEL_FILE_PATH)
    tmp = tempfile.NamedTemporaryFile()
    
    tmp_file = open(tmp.name, "wb")
    object.download_fileobj(tmp_file)
    
    # load model
    model = tf.keras.models.load_model(tmp.name)
    
    # make prediction
    prediction_matrix = model.predict(image_np)
    
    # define classes
    classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    
    # get results
    prediction_matrix = list(prediction_matrix)
    max_val = max(prediction_matrix)
    index = prediction_matrix.index(max_val)
    prediction = classes[index]
    
    tmp_file.close()
    
    body = {
        "prediction": prediction
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Access-Control-Allow-Origin": '*',
            "Access-Control-Allow-Credentials": True,
        }
    }

    return response
