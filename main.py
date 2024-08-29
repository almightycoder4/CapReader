import io
import json
import requests
from io import BytesIO
from time import time
import base64
import cgi
import cv2
import numpy as np
import os

from torchOcr import OCRModel

def validate_image_url(img_url):
    response = requests.get(img_url)
    if response.status_code != 200:
        raise ValueError("Failed to retrieve image from URL")
    if response.headers['Content-Type'] not in ['image/jpeg', 'image/jpg', 'image/png']:
        raise ValueError("Invalid file type")
    return response.content

def enhance_image(img_buffer, brightness, contrast, sharpness):
    # Convert image buffer to numpy array
    image_np = np.frombuffer(img_buffer.getvalue(), dtype=np.uint8)
    
    # Decode the image using OpenCV
    img = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    
    # Resize image to 200x50
    img_resized = cv2.resize(img, (200, 50), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding (e.g., Otsu's thresholding)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Track if any enhancement is applied
    enhancement_applied = False

    # Optionally adjust brightness, contrast, and sharpness
    if brightness != 1.0 or contrast != 1.0:
        thresh = cv2.convertScaleAbs(thresh, alpha=contrast, beta=brightness)
        enhancement_applied = True
    
    # Sharpness can be adjusted using a kernel
    if sharpness != 1.0:
        kernel = np.array([[-1, -1, -1], [-1, 9 * sharpness, -1], [-1, -1, -1]])
        thresh = cv2.filter2D(thresh, -1, kernel)
        enhancement_applied = True

    # Save enhanced image locally if any enhancement was applied
    # if enhancement_applied:
    #     output_path = 'enhanced_image.png'  # Local path where the image will be saved
    #     cv2.imwrite(output_path, thresh)
        print(f"Enhanced image saved to {output_path}")

    # Encode the processed image back to a buffer
    _, buffer = cv2.imencode('.png', thresh)
    return BytesIO(buffer)

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

        # Initialize variables
        img_buffer = None
        img_url = None
        brightness = 1.0
        contrast = 1.0
        sharpness = 1.0

        if 'multipart/form-data' in content_type:
            # Handle file upload via form-data
            file_content = base64.b64decode(event['body'])  # Decode the base64 form-data
            boundary = content_type.split(';')[1].split('=')[1].strip()

            # Parse the form-data
            environ = {'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
            form_data = cgi.FieldStorage(
                fp=BytesIO(file_content),
                environ=environ,
                keep_blank_values=True
            )
            
            file_item = form_data['file']  # This should match the key name in Postman
            img_buffer = BytesIO(file_item.file.read())  # Read the image file content
        else:
            # Handle JSON input
            body = json.loads(event.get('body', '{}'))
            img_url = body.get('imgUrl')
            base64_img = body.get('base64Image')
            brightness = body.get('brightness', 1.0)
            contrast = body.get('contrast', 1.0)
            sharpness = body.get('sharpness', 1.0)

            if img_url:
                img_content = validate_image_url(img_url)
                img_buffer = io.BytesIO(img_content)
            elif base64_img:
                img_buffer = BytesIO(base64.b64decode(base64_img))

        if not img_buffer:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Either imgUrl, base64Image, or image buffer must be provided"})
            }

        if path == '/captchaSolver' and http_method == 'POST':
            # Enhance image before OCR
            enhanced_img_buffer = enhance_image(img_buffer, brightness, contrast, sharpness)
            
            # Predict using the OCR model
            detected_text, confidenceScore = ocr_model.predict(enhanced_img_buffer)
            result_message = "OCR Completed Successfully."
        else:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "Path not found or Invalid request type."})
            }

        end_time = time()
        execution_time = end_time - start_time

        return {
            "statusCode": 200,
            "body": json.dumps({
                "detected_text": detected_text,
                "confidence_score": confidenceScore,
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