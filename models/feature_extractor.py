# models/feature_extractor.py
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import os 

class YOLOFeatureExtractor:
    """Extracts features using YOLOv8 backbone."""

    def __init__(self):
        self.model = YOLO('yolov8n.pt').model 
        self.backbone = self.model.model[:10] # First 10 layers
        self.backbone.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Dynamically determine the output feature dimension
        # Create a dummy tensor to pass through the backbone and get its output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, 3 channels, 224x224
            dummy_output = self.backbone(dummy_input)
            self._feature_dim = dummy_output.flatten().shape[0] # Get the flattened dimension
        print(f"YOLOv8n backbone will produce features of dimension: {self._feature_dim}")


    def get_features(self, image_path):
        """
        Convert image to 1D feature vector.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                features = self.backbone(img_tensor).flatten()
            
            return features.numpy().astype('float32')
        except Exception as e:
            raise ValueError(f"Error processing {image_path}: {str(e)}")

    @property
    def feature_dim(self):
        """Returns the expected dimension of the extracted features."""
        return self._feature_dim