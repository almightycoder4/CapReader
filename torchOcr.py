import torch
from torchvision import transforms as T
from PIL import Image

class OCRModel:
    def __init__(self):
        # Load the model
        self.model = torch.hub.load('./torch/hub/baudm_parseq_main', 'parseq', source='local', pretrained=True, trust_repo=True).eval()
        # self.model = torch.hub.load('./TorchOCR/hub/baudm_parseq_main', 'custom', path='/TorchOCR/hub/checkpoints/parseq-bb5792a6.pt', source='local') 

        # Preprocess transformation
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

    def predict(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = self._preprocess(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            pred = self.model(image).softmax(-1)
            label, _ = self.model.tokenizer.decode(pred)
            print(_)
        
        return label[0]

# Example usage
if __name__ == '__main__':
    ocr_model = OCRModel()  # Instantiate the class
    result = ocr_model.predict('../captcha_datasets/6mfM.png')  # Call the predict method on the instance
    print("Detected Text:", result)