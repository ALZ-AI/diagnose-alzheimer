import json
import tensorflow as tf
import boto3
import tempfile
from cgi import parse_header, parse_multipart
import base64
import io
from PIL import Image
import numpy as np

BUCKET_NAME = "diagnose-alzheimer-bucket"
MODEL_FILE_PATH = 'outs/model.h5'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_PATH

def predict(event, context):
    
    # check get or post
    if event["requestContext"]["http"]["method"] == "GET":
        return {
            "statusCode": 200,
            "body": "Hello World!"
        }
    
    c_type, c_data = parse_header(event['headers']['content-type'])
    assert c_type == 'multipart/form-data'
    decoded_string = base64.b64decode(event['body'])
    c_data['boundary'] = bytes(c_data['boundary'], "utf-8")
    c_data['CONTENT-LENGTH'] = event['headers']['content-length']
    form_data = parse_multipart(io.BytesIO(decoded_string), c_data)
    
    image_str = form_data["image"][0]
    image = Image.open(io.BytesIO(image_str))
    image = np.array(image)
    
    # download model from S3 Bucket to lambda
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(BUCKET_NAME)
    object = bucket.Object(MODEL_FILE_PATH)
    tmp = tempfile.NamedTemporaryFile()
    
    tmp_file = open(tmp.name, "wb")
    object.download_fileobj(tmp_file)
    
    # load model
    model = tf.keras.models.load_model(tmp_file)
    
    # make prediction
    prediction = model.predict(image)
    
    tmp_file.close()
    
    body = {
        "prediction": prediction
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
