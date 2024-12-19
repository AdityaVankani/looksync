import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Get list of actors (subdirectories) in 'data1' folder, excluding hidden files
actors = [actor for actor in os.listdir('data1') if not actor.startswith('.')]

filenames = []

for actor in actors:
    actor_path = os.path.join('data1', actor)
    if os.path.isdir(actor_path):  # Ensure it's a directory
        for file in os.listdir(actor_path):
            # Check if file is not hidden (e.g., .DS_Store) and is an image file
            if not file.startswith('.') and (file.endswith('.jpg') or file.endswith('.png')):
                filenames.append(os.path.join(actor_path, file))

# Save filenames to a pickle file
# pickle.dump(filenames, open('filenames1.pkl', 'wb'))

# Load the saved filenames
filenames = pickle.load(open('filenames1.pkl', 'rb'))

# Initialize the ResNet50 model from torchvision (pre-trained on ImageNet)
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer (use as feature extractor)
model.eval()  # Set the model to evaluation mode (important for feature extraction)

# Define image preprocessing transformations (similar to VGGFace preprocessing)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# Function to extract features from an image using the ResNet50 model
def feature_extractor(img_path, model):
    img = Image.open(img_path).convert('RGB')  # Load the image and ensure it's RGB
    img_preprocessed = preprocess(img)  # Apply transformations
    img_tensor = img_preprocessed.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # Disable gradient calculation (faster inference)
        features = model(img_tensor).flatten().cpu().numpy()  # Extract features and convert to numpy array
        print(len(features))
    return features

# Extract features for all images
features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file, model))
    break

# Save extracted features to a pickle file
# pickle.dump(features, open('embedding1.pkl', 'wb'))