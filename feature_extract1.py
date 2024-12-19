import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Initialize model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# Function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    print(len(result))
    normalized_result = result / norm(result)
    return normalized_result

# Collect filenames with specific image file extensions
valid_extensions = {'.jpeg', '.jpg', '.png'}
filenames = [os.path.join('images', file) for file in os.listdir('images')
             if os.path.splitext(file)[1].lower() in valid_extensions]

print(filenames)

# Extract features for each image and store in a list
feature_list = []
for file in tqdm(filenames):
    print(file)
    feature_list.append(extract_features(file, model))
    break

# Save features and filenames with pickle
# pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
# pickle.dump(filenames, open('filenames.pkl', 'wb'))