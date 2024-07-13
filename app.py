import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model('cat_dog_model.h5')

def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((100, 100))
    image = np.array(image) / 255.0
    return image.reshape(1, 100, 100, 3)

st.title("Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    if st.button("Classify"):
        processed_image = preprocess_image(image)
        y_pred = model.predict(processed_image)
        pred = 'Cat' if y_pred > 0.5 else 'Dog'
        st.write(f"Model says it is a: {pred}")
