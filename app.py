import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Eye Disease Classification", page_icon="👁️")

CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
LABELS = {
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "normal": "Normal",
}


@st.cache_resource
def get_model():
    return load_model("model.keras", compile=False)


st.title("Deep Learning-Based Multi-Class Eye Disease Classification using CNN")
st.write(
    "Upload an eye image. A fine-tuned MobileNetV2 CNN predicts one of four classes: "
    "Normal, Diabetic Retinopathy, Cataract or Glaucoma."
)
st.info("College mini project. Screening aid only, not a medical diagnosis.")

file = st.file_uploader("Upload a fundus / eye image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    arr = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), axis=0)  # raw 0-255, model rescales inside
    with st.spinner("Predicting..."):
        probs = get_model().predict(arr, verbose=0)[0]
    best = int(np.argmax(probs))
    if probs[best] < 0.6:
        st.warning("Uncertain. Please upload a clear fundus image.")
    else:
        st.success(f"Prediction: {LABELS[CLASS_NAMES[best]]} ({probs[best] * 100:.1f}% confidence)")
    st.subheader("Class probabilities")
    for i, name in enumerate(CLASS_NAMES):
        st.write(LABELS[name])
        st.progress(float(probs[i]))
