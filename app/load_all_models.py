"""
Complete Model Loading Script for Deepfake Detection
Loads all three trained models: CustomCNN, EfficientNet, and MobileNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import io
from torchvision import transforms

# ==============================================================================
# 1. CUSTOM CNN ARCHITECTURE (Exact copy from training)
# ==============================================================================
class CustomCNN(nn.Module):
    """
    Custom CNN architecture for deepfake detection.
    Trained on OpenFake dataset with binary classification (Real vs Fake).
    """
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Regularization
        self.dropout = nn.Dropout(0.5)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification output

    def forward(self, x):
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Returns logits
        return x

# ==============================================================================
# 2. MODEL LOADING FUNCTIONS
# ==============================================================================
def load_custom_cnn_model(model_path="models/base_cnn_complete.pth", device="cpu"):
    """
    Load the trained CustomCNN model.

    Args:
        model_path (str): Path to the saved model file
        device (str): Device to load model on ('auto', 'cpu', 'cuda')

    Returns:
        CustomCNN: Loaded model in evaluation mode
    """
    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # For Streamlit deployment, force CPU
    device = torch.device("cpu")

    print(f"🔄 Loading CustomCNN model from {model_path}...")

    # Fix for pickle loading: add CustomCNN to __main__ module
    import sys
    sys.modules['__main__'].CustomCNN = CustomCNN

    # Initialize model
    model = CustomCNN()

    # Load saved weights
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Handle different save formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is the state_dict directly
            model.load_state_dict(checkpoint)
    elif hasattr(checkpoint, 'state_dict'):
        model.load_state_dict(checkpoint.state_dict())
    else:
        # Fallback: assume it's a state_dict dict
        model.load_state_dict(checkpoint)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Convert to half precision for memory optimization (works on both CPU and GPU)
    model = model.half()

    print(f"✅ CustomCNN loaded successfully on {device} (FP16)!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model

def load_efficientnet_model(model_path="models/efficientnet/efficientnet_complete", device="cpu"):
    """
    Load the trained EfficientNet model.

    Args:
        model_path (str): Path to the saved model directory
        device (str): Device to load model on ('auto', 'cpu', 'cuda')

    Returns:
        tuple: (model, processor)
    """
    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # For Streamlit deployment, force CPU
    device = torch.device("cpu")

    print(f"🔄 Loading EfficientNet model from {model_path}...")

    # Load model and processor
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Convert to half precision for memory optimization (works on both CPU and GPU)
    model = model.half()

    print(f"✅ EfficientNet loaded successfully on {device} (FP16)!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, processor

def load_mobilenet_model(model_path="models/mobilenet/mobilenet_complete", device="cpu"):
    """
    Load the trained MobileNet model.

    Args:
        model_path (str): Path to the saved model directory
        device (str): Device to load model on ('auto', 'cpu', 'cuda')

    Returns:
        tuple: (model, processor)
    """
    # Determine device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # For Streamlit deployment, force CPU
    device = torch.device("cpu")

    print(f"🔄 Loading MobileNet model from {model_path}...")

    # Load model and processor
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Convert to half precision for memory optimization (works on both CPU and GPU)
    model = model.half()

    print(f"✅ MobileNet loaded successfully on {device} (FP16)!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, processor

def load_all_models(device="cpu"):
    """
    Load all three trained models.

    Args:
        device (str): Device to load models on ('cpu' for Streamlit deployment)

    Returns:
        dict: Dictionary containing all loaded models and processors
    """
    print("🚀 Loading all deepfake detection models (optimized for CPU/FP16)...\n")

    models = {}

    # Load Custom CNN
    models['custom_cnn'] = load_custom_cnn_model(device=device)

    # Load EfficientNet
    models['efficientnet'], models['efficientnet_processor'] = load_efficientnet_model(device=device)

    # Load MobileNet
    models['mobilenet'], models['mobilenet_processor'] = load_mobilenet_model(device=device)

    print("\n🎉 All models loaded successfully!")
    return models

# ==============================================================================
# 3. INFERENCE FUNCTIONS
# ==============================================================================
def predict_with_custom_cnn(model, image_tensor):
    """
    Make prediction with CustomCNN model.

    Args:
        model: Loaded CustomCNN model
        image_tensor: Preprocessed image tensor (3, 224, 224)

    Returns:
        dict: Prediction results
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.unsqueeze(0).to(device).half()  # Add batch dimension and convert to FP16

    with torch.no_grad():
        logits = model(image_tensor)
        prob_fake = torch.sigmoid(logits).item()
        prob_real = 1 - prob_fake
        prediction = "FAKE" if prob_fake > 0.5 else "REAL"

    return {
        "model": "CustomCNN",
        "prediction": prediction,
        "fake_probability": prob_fake,
        "real_probability": prob_real
    }

def predict_with_pretrained_model(model, processor, image, model_name):
    """
    Make prediction with pretrained model (EfficientNet/MobileNet).

    Args:
        model: Loaded pretrained model
        processor: Image processor
        image: PIL Image
        model_name: Name of the model for logging

    Returns:
        dict: Prediction results
    """
    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Convert inputs to half precision to match model
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].half()

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    fake_prob = probs[0][1].item()
    real_prob = probs[0][0].item()
    prediction = "FAKE" if fake_prob > 0.5 else "REAL"

    return {
        "model": model_name,
        "prediction": prediction,
        "fake_probability": fake_prob,
        "real_probability": real_prob
    }

def predict_image(image_input, models_dict):
    """
    Predict if an image is real or fake using all three models.

    Args:
        image_input: PIL Image, image bytes, or file path
        models_dict: Dictionary with loaded models from load_all_models()

    Returns:
        dict: Predictions from all models
    """
    # Load image based on input type
    if isinstance(image_input, str):
        # File path
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, bytes):
        # Bytes
        image = Image.open(io.BytesIO(image_input)).convert('RGB')
    elif hasattr(image_input, 'convert'):
        # PIL Image
        image = image_input.convert('RGB')
    else:
        raise ValueError("image_input must be a file path (str), bytes, or PIL Image")

    results = {}

    # Custom CNN prediction (needs tensor preprocessing) - only if available
    if 'custom_cnn' in models_dict:
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = base_transform(image)
        results['custom_cnn'] = predict_with_custom_cnn(models_dict['custom_cnn'], image_tensor)

    # EfficientNet prediction - only if available
    if 'efficientnet' in models_dict:
        results['efficientnet'] = predict_with_pretrained_model(
            models_dict['efficientnet'],
            models_dict['efficientnet_processor'],
            image,
            "EfficientNet"
        )

    # MobileNet prediction - only if available
    if 'mobilenet' in models_dict:
        results['mobilenet'] = predict_with_pretrained_model(
            models_dict['mobilenet'],
            models_dict['mobilenet_processor'],
            image,
            "MobileNet"
        )

    return results

# ==============================================================================
# 4. EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    print("🧪 Testing model loading and inference...")

    # Load all models
    models = load_all_models()

    # Test with dummy image (you can replace with actual image)
    # For testing, we'll create a simple colored square
    test_image = Image.new('RGB', (224, 224), color='gray')

    # Make predictions
    results = predict_image(test_image, models)

    # Print results
    print("\n📊 PREDICTION RESULTS:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"{model_name.upper()}: {result['prediction']} "
              f"(Fake: {result['fake_probability']:.3f})")

    print("\n✅ All models working correctly!")