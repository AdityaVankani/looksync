# **LookSync: Personalized Product Recommendation System**  

LookSync is an advanced product recommendation system that combines the power of deep learning and transfer learning to provide personalized suggestions. The system also includes a fun feature to identify the Bollywood celebrity you resemble the most!  

## **Features**  
1. **Product Recommendation**  
   - Provides personalized product recommendations based on user preferences using similarity matching.  

2. **Celebrity Look-Alike Finder**  
   - Matches user-uploaded photos to Bollywood celebrity images using deep learning.  

---

## **Technologies Used**  

- **Programming Language**: Python  
- **Deep Learning Framework**: PyTorch  
- **Model Architecture**: ResNet-50 (pre-trained on ImageNet)  
- **Web Framework**: Streamlit  
- **Data Storage**: `.pkl` files for embedding storage  
- **Libraries**: NumPy, Pandas, OpenCV, scikit-learn for data handling and processing  

---

## **Methodology**  

### **Task 1: Product Recommendation**  
1. **Model**: ResNet-50 pre-trained on ImageNet was used for feature extraction.  
2. **Embeddings**: Feature embeddings for products were generated and stored in a `.pkl` file for fast comparisons.  
3. **Similarity Matching**:  
   - Top 5 items are retrieved using similarity matching with a threshold of **0.75**.  
   - Ensures relevant and accurate recommendations.  
4. **Interface**: A user-friendly interface built with **Streamlit** allows easy interaction, including image uploads and result displays.  

### **Task 2: Celebrity Look-Alike Finder**  
1. **Model**: ResNet-50 generates feature embeddings for Bollywood celebrity images.  
2. **Embeddings Storage**:  
   - Embeddings for all celebrity images are stored in a `.pkl` file, enabling fast similarity searches.  
3. **Similarity Search**:  
   - Compares user-uploaded images with stored embeddings to identify the closest celebrity match.  
4. **Interface**: **Streamlit** powers a smooth and intuitive interface for effortless uploads and result display.  

---
## **Dataset used**
1.**Fashion Product Images Dataset**:
It consist of images of several different products( around 44k images ).
Link:https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset.

2.**Bollywood celeb localized face dataset**:
This dataset contains the localized face of 100 Bollywood celebrities. every class has samples between 80 to 150 of size 64 X 64 pixels. samples contain wild conditions such as different orientations, illuminations, age transitions, etc
Link:https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset

## **Installation and Setup**

### **Steps to Run**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/AdityaVankani/LookSync.git  
   cd LookSync
2. run extract_feature and extract_feature1
3. Run final.py
   ```bash
   streamlit run final.py
