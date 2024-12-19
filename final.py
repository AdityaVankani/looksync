import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import torch
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# Initialize session state for page tracking
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Load models and embeddings
feature_list_fashion = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames_fashion = pickle.load(open('filenames.pkl', 'rb'))

feature_list_celebrity = pickle.load(open('embedding1.pkl', 'rb'))
filenames_celebrity = pickle.load(open('filenames1.pkl', 'rb'))

# Load models
fashion_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
fashion_model.trainable = False
fashion_model = tensorflow.keras.Sequential([fashion_model, GlobalMaxPooling2D()])

detector = MTCNN()
celebrity_model = models.resnet50(pretrained=True)
celebrity_model.fc = torch.nn.Identity()
celebrity_model.eval()

# Helper function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"File upload failed: {e}")
        return 0

# Feature extraction for fashion recommender
def feature_extraction_fashion(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Feature extraction for celebrity look-alike
def extract_features_celebrity(img_path, model, detector):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect(img_rgb)
    if results[0] is not None:
        x, y, width, height = results[0][0]
        face = img_rgb[int(y):int(y + height), int(x):int(x + width)]
        face_img = Image.fromarray(face).resize((224, 224))
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        face_tensor = preprocess(face_img).unsqueeze(0)
        with torch.no_grad():
            result = model(face_tensor).flatten().numpy()
        return result
    else:
        return None

# Recommendation functions
def recommend_fashion(features, feature_list, threshold=0.70):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    print(distances)
    matches = []
    for i, distance in zip(indices[0], distances[0]):
        if distance < threshold:  
            matches.append(i)
    
    return matches if matches else None

def recommend_celebrity(feature_list, features, threshold=30):
    similarity = []
    print(len(features))
    for i in range(len(feature_list)):
        distance = np.linalg.norm(features - feature_list[i])
        similarity.append(distance)
    min_distance = min(similarity)
    if min_distance <= threshold:
        index_pos = sorted(list(enumerate(similarity)), key=lambda x: x[1])[0][0]
        return index_pos
    else:
        return None

def home_page():
    st.markdown("""
        <style>
            .welcome-title {
                font-size: 2.5rem;
                font-weight: 600;
                text-align: center;
                color: #ff6f61;
            }
            .welcome-text {
                font-size: 1.2rem;
                line-height: 1.6;
                color: grey;
                margin-top: 20px;
                text-align: justify;
            }
            .options-heading {
                font-size: 1.8rem;
                font-weight: bold;
                color: #4a90e2;
                margin-top: 40px;
                text-align: center;
            }
            .option-list {
                font-size: 1.1rem;
                color: grey;
                margin-top: 10px;
                padding-left: 30px;
                list-style-type: disc;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="welcome-title">Welcome to Your Personal Recommendation Hub! ✨</p>', unsafe_allow_html=True)

    st.markdown("""
        <p class="welcome-text">
            Dive into a world of style and discovery with our dual-purpose recommendation system! 
            Created to offer you personalized insights and a touch of fun, this app transforms the way 
            you explore trends and discover celebrity look-alikes. Whether you're here for fashion inspiration 
            or a bit of celebrity comparison, you've come to the right place.
        </p>
        
        <p class="options-heading">Choose Your Journey:</p>
        <ul class="option-list">
            <li><strong>Fashion Recommender:</strong> Upload an image of a style you admire, and our system will match it with similar fashion pieces, curated for your unique taste.</li>
            <li><strong>Celebrity Look-Alike:</strong> Upload your own photo, and we'll show you the celebrity you most resemble — perfect for fun or satisfying that "who do I look like?" curiosity!</li>
        </ul>
        
        <p class="welcome-text" style="text-align: center; margin-top: 30px;">
            <em>Use the navigation bar to get started on your journey!</em>
        </p>
    """, unsafe_allow_html=True)

def fashion_recommender_page():
    st.title("Fashion Recommender")
    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Uploaded Image")
            features = feature_extraction_fashion(os.path.join("uploads", uploaded_file.name), fashion_model)
            indices = recommend_fashion(features, feature_list_fashion)

            if indices:
                cols = st.columns(min(len(indices), 5))  
                for i, col in enumerate(cols):
                    if i < len(indices):  
                        with col:
                            st.image(filenames_fashion[indices[i]])
            else:
                st.error("No match found!")

# Celebrity Look-Alike Page
def celebrity_lookalike_page():
    st.title("Celebrity Look-Alike")
    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Uploaded Image")
            features = extract_features_celebrity(os.path.join("uploads", uploaded_file.name), celebrity_model, detector)
            
            if features is not None:
                index_pos = recommend_celebrity(feature_list_celebrity, features)
                if index_pos is not None:
                    predicted_actor = os.path.basename(filenames_celebrity[index_pos]).split('_')[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.header('Uploaded Image')
                        st.image(display_image)
                    with col2:
                        st.header(f"Looks like {predicted_actor}")
                        st.image(filenames_celebrity[index_pos], width=300)
                else:
                    st.error("No match found!")
            else:
                st.error("No face detected. Please upload a clearer image.")

# Navbar
st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("Fashion Recommender"):
    st.session_state.page = "Fashion Recommender"
if st.sidebar.button("Celebrity Look-Alike"):
    st.session_state.page = "Celebrity Look-Alike"

# Render pages based on navigation selection
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Fashion Recommender":
    fashion_recommender_page()
elif st.session_state.page == "Celebrity Look-Alike":
    celebrity_lookalike_page()