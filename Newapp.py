import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # ✅ Make sure this matches your saved file
    return model

model = load_model()

st.title("🧠 Real vs AI-Generated Image Detector")
st.write("Upload an image to classify it as **Real** or **AI-generated (Fake)**.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    img = image.resize((32, 32))               # ✅ Resize
    img = np.array(img) / 255.0                # ✅ Normalize
    img = np.expand_dims(img, axis=0)          # ✅ Add batch dimension
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    class_names = ['Fake Image', 'Real Image']
    
    st.markdown(f"### Prediction: **{class_names[predicted_class]}**")
    st.markdown(f"Confidence: `{confidence:.2f}%`")
