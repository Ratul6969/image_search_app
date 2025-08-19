# models/efficientnet_extractor.py
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class EfficientNetFeatureExtractor:
    """Extracts features using a pre-trained EfficientNet backbone."""

    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        self.model_name = model_name

        # Load the pre-trained EfficientNet model
        self.model = getattr(models, model_name)(pretrained=pretrained)
        
        # Remove the classification head (last layer of the model)
        # EfficientNet's backbone typically ends before the final classifier.
        # The list(self.model.children()) gives us the layers.
        # [:-1] removes the very last layer (the classifier).
        self.backbone = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        # Ensure the model is in evaluation mode; important for consistent inference
        self.backbone.eval() 

        # Define the image preprocessing pipeline.
        # This matches the standard transformations for models pre-trained on ImageNet.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize image to 224x224 pixels
            transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor (HWC -> CHW, 0-255 -> 0-1)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
        ])

        # Dynamically determine the output feature dimension of the backbone.
        # We pass a dummy input through the backbone to get its output shape,
        # then flatten it to get the final 1D feature vector dimension.
        with torch.no_grad(): # Disable gradient calculation for efficiency during this dummy pass
            dummy_input = torch.randn(1, 3, 224, 224) # A single dummy image: batch size 1, 3 color channels, 224x224 pixels
            dummy_output = self.backbone(dummy_input)
            self._feature_dim = dummy_output.flatten().shape[0] # Flatten the output and get its size
        print(f"{model_name} backbone will produce features of dimension: {self._feature_dim}")


    def get_features(self, image_path):
        """
        Converts an image from a given path into a 1D feature vector using the EfficientNet backbone.
        
        Args:
            image_path (str): The file path to the image.
            
        Returns:
            np.ndarray: A 1D NumPy array representing the image's feature vector (float32).
            
        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If there's an error during image processing or feature extraction.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        try:
            img = Image.open(image_path).convert('RGB') # Open image and ensure it's in RGB format
            img_tensor = self.transform(img).unsqueeze(0) # Apply transformations and add a batch dimension
            
            with torch.no_grad(): # Perform inference without tracking gradients
                features = self.backbone(img_tensor).flatten() # Pass through backbone and flatten to 1D
            
            # Convert PyTorch tensor to NumPy array and ensure float32 dtype for Annoy compatibility
            return features.numpy().astype('float32')
        except Exception as e:
            raise ValueError(f"Error processing {image_path}: {str(e)}")

    @property
    def feature_dim(self):
        """Returns the expected dimension of the extracted features."""
        return self._feature_dim