import json
from main import lambda_handler

event = {
    "headers": {
        "Content-Type": "application/json"
    },
    "body": json.dumps({
        "imgUrl": "https://iduploadbucket.s3.ap-south-1.amazonaws.com/DFaqQf.png",
        "brightness":1,
        "contrast":4,
        "saturation":4
    }),
    "httpMethod": "POST",
    "isBase64Encoded": False,
    "path": "/captchaSolver"
}

response = lambda_handler(event, None)
print(json.dumps(response, indent=4))

