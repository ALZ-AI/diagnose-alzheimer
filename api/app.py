import json
import tensorflow as tf

def predict(event, context):
    body = {
        "message": "Hello, world! Your function executed successfully!",
        "tf_version": tf.__version__
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
