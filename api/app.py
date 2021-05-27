import tensorflow as tf
import boto3
import tempfile
import base64
import io
from PIL import Image, ImageFile
import numpy as np
from lambda_decorators import cors_headers, load_json_body, json_http_resp


BUCKET_NAME = "diagnose-alzheimer-bucket"
MODEL_FILE_PATH = 'outs/model.h5'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_PATH

ImageFile.LOAD_TRUNCATED_IMAGES = True

@cors_headers
@json_http_resp
@load_json_body
def predict(event, context):
    
    # get params
    data = event["body"]
    
    # convert base64 string to pillow image
    decoded_string = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(decoded_string)).resize((256, 256)).convert("L")
    
    # convert pillow image to numpy array
    image_np = np.array(image)
    
    # prepare for prediction
    image_np = np.array([image_np.reshape((256, 256, 1))])
    
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
    
    response = {
        "prediction": prediction
    }
    
    return response
