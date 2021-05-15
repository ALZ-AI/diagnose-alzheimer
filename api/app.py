import json
import torch

def predict(event, context):
    body = {
        "message": "Hello, world! Your function executed successfully!",
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
