from flask import Flask, request, jsonify, send_file
import io
import json
import requests
import base64
import cgi
import cv2
import numpy as np
from io import BytesIO
from time import time
from torchOcr import OCRModel

app = Flask(__name__)

def validate_image_url(img_url):
    response = requests.get(img_url)
    if response.status_code != 200:
        raise ValueError("Failed to retrieve image from URL")
    if response.headers['Content-Type'] not in ['image/jpeg', 'image/jpg', 'image/png']:
        raise ValueError("Invalid file type")
    return response.content

def enhance_image(img_buffer, brightness, contrast, sharpness):
    # Convert image buffer to numpy array
    print(brightness, contrast, sharpness, "image config")
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
    enhancement_applied = True

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
    if enhancement_applied:
        output_path = 'enhanced_image.png'  # Local path where the image will be saved
        cv2.imwrite(output_path, thresh)
        print(f"Enhanced image saved to {output_path}")

    # Encode the processed image back to a buffer
    _, buffer = cv2.imencode('.png', thresh)
    return BytesIO(buffer)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Hello from CaptchaSolver v1.0!"})

@app.route('/captchaSolver', methods=['POST'])
def captcha_solver():
    start_time = time()

    ocr_model = OCRModel()

    try:
        content_type = request.headers.get('Content-Type', '')

        # Initialize variables
        img_buffer = None
        img_url = None

        if 'multipart/form-data' in content_type:
            # Handle file upload via form-data
            file_item = request.files['file']  # This should match the key name in Postman
            img_buffer = BytesIO(file_item.read())  # Read the image file content

            # Retrieve brightness, contrast, and sharpness from form data
            brightness = float(request.form.get('brightness', 1.0))
            contrast = float(request.form.get('contrast', 1.0))
            sharpness = float(request.form.get('sharpness', 1.0))

        else:
            # Handle JSON input
            data = request.get_json()
            img_url = data.get('imgUrl')
            base64_img = data.get('base64Image')
            brightness = data.get('brightness', 1.0)
            contrast = data.get('contrast', 1.0)
            sharpness = data.get('sharpness', 1.0)

            if img_url:
                img_content = validate_image_url(img_url)
                img_buffer = io.BytesIO(img_content)
            elif base64_img:
                img_buffer = BytesIO(base64.b64decode(base64_img))

        if not img_buffer:
            return jsonify({"error": "Either imgUrl, base64Image, or image buffer must be provided"}), 400

        # Enhance image before OCR
        enhanced_img_buffer = enhance_image(img_buffer, brightness, contrast, sharpness)
        
        # Predict using the OCR model
        detected_text, confidence_score = ocr_model.predict(enhanced_img_buffer)
        result_message = "OCR Completed Successfully."

        end_time = time()
        execution_time = end_time - start_time

        return jsonify({
            "detected_text": detected_text,
            "confidence_score": confidence_score,
            "result": result_message,
            "execution_time": f"{round(execution_time, 2)} sec",
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
