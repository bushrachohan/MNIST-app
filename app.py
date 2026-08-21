import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.datasets import mnist
from PIL import Image


# Load the trained model
def load_trained_model():
    return load_model("mnist_ann.h5")

# Preprocess uploaded image
def preprocess_image(image):
    image = Image.open(image).convert('L').resize((28, 28))  # Convert to grayscale and resize
    image = img_to_array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Reshape for model
    return image

# Streamlit UI
def main():
    st.title("MNIST Digit Recognition")
    st.write("Upload a digit image (28x28 grayscale) to classify it.")
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)
        
        model = load_trained_model()
        processed_image = preprocess_image(uploaded_file)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        
        st.write(f"Predicted Digit: {predicted_class}")

if __name__ == "__main__":
    main()
