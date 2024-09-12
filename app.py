import streamlit as st
import os
from PIL import Image
from model import load_model, predict_image

# Load the ResNet50 model
model = load_model()

# Title of the app
st.title("Image Classifier using ResNet50")

# Upload image
uploaded_file = st.file_uploader("Upload an RGB Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded image to the 'images' folder (optional)
    image_path = os.path.join("images", uploaded_file.name)
    
    # Ensure the 'images' directory exists
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Save the uploaded file
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predict the class
    st.write("Classifying...")
    predictions = predict_image(model, image_path)

    # Display the predictions
    for pred in predictions:
        st.write(f"{pred[1]}: {round(pred[2] * 100, 2)}%")
