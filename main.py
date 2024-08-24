import io
import json
import requests
from io import BytesIO
from time import time
import base64

from torchOcr import OCRModel

def validate_image_url(img_url):
    response = requests.get(img_url)
    if response.status_code != 200:
        raise ValueError("Failed to retrieve image from URL")
    if response.headers['Content-Type'] not in ['image/jpeg', 'image/jpg', 'image/png']:
        raise ValueError("Invalid file type")
    return response.content

def lambda_handler(event, context):
    http_method = event['httpMethod']
    path = event['path']
    start_time = time()

    ocr_model = OCRModel()

    try:
        if http_method == 'GET' and path == '/':
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Hello from CaptchaSolver v1.0!"})
            }
        
        if http_method != 'POST':
            return {
                "statusCode": 405,
                "body": json.dumps({"error": "Method not allowed"})
            }

        content_type = event['headers'].get('Content-Type', '')

        if 'multipart/form-data' in content_type:
            # Handle file upload via Postman
            file_content = event['body']
            img_buffer = BytesIO(base64.b64decode(file_content))
            img_url = None  # No URL provided in this case
            brightness = body.get('brightness', 1.0)
            contrast = body.get('contrast', 1.0)
            sharpness = body.get('sharpness', 1.0)
        else:
            # Handle JSON input
            body = json.loads(event.get('body', '{}'))
            img_url = body.get('imgUrl')
            img_buffer = None
            brightness = body.get('brightness', 1.0)
            contrast = body.get('contrast', 1.0)
            sharpness = body.get('sharpness', 1.0)

        if not img_url and not img_buffer:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Either imgUrl or image buffer must be provided"})
            }

        if img_url:
            img_content = validate_image_url(img_url)
            image_buffer = io.BytesIO(img_content)
        else:
            image_buffer = img_buffer

        if path == '/captchaSolver':
            detected_text = ocr_model.predict(image_buffer, brightness, contrast, sharpness)
            result_message = "OCR Completed Successfully."
        else:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "Path not found"})
            }

        end_time = time()
        execution_time = end_time - start_time

        return {
            "statusCode": 200,
            "body": json.dumps({
                "detected_text": detected_text,
                "result": result_message,
                "execution_time": f"{round(execution_time, 2)} sec",
            })
        }

    except ValueError as ve:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(ve)})
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error", "details": str(e)})
        }
