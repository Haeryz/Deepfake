import torch
import torch.nn as nn
import torch.nn.functional as F

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

def load_custom_cnn_model(model_path="models/base_cnn_complete.pth", device="auto"):
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

    print(f"🔄 Loading CustomCNN model from {model_path}...")

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

    print(f"✅ Model loaded successfully on {device}!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model

# Example usage
if __name__ == "__main__":
    # Load model
    model = load_custom_cnn_model()
    device = next(model.parameters()).device  # Get model's device

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        probability = torch.sigmoid(output).item()
        prediction = "FAKE" if probability > 0.5 else "REAL"

    print(f"🧪 Test prediction: {prediction} (Fake probability: {probability:.3f})")