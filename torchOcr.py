import os
import torch
from torchvision import transforms as T
from PIL import Image, ImageEnhance
from io import BytesIO

MODEL_PATH = '/var/task/torch/hub/baudm_parseq_main'
class OCRModel:
    def __init__(self):
        # Load the model
        print(MODEL_PATH)
        self.model = torch.hub.load(MODEL_PATH, 'parseq', source='local', pretrained=True, trust_repo=True).eval()

        # Preprocess transformation
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

    def adjust_image(self, image, brightness=1.0, contrast=1.0, sharpness=1.0):
        """
        Adjust the brightness, contrast, and sharpness of the image.
        """
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)

        return image

    def predict(self, image_input, brightness=1.0, contrast=1.0, sharpness=1.0):
        """
        Predict text from an image. The image can be provided as a file path or a buffer.
        """
        if isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert('RGB')
        else:
            image = Image.open(image_input).convert('RGB')

        # Adjust the image according to user-defined values
        image = self.adjust_image(image, brightness, contrast, sharpness)
        # Preprocess the image
        image = self._preprocess(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            pred = self.model(image).softmax(-1)
            label, _ = self.model.tokenizer.decode(pred)
            tensor_values = torch.tensor(_[0])
            confidenceScore = round(tensor_values[-1].item(), 3)
            print(_[0])

        
        return label[0], confidenceScore
