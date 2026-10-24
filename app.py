import os

import numpy as np
import streamlit as st
from PIL import Image

import ui

st.set_page_config(page_title="Eye Disease Classification", page_icon="👁️", layout="centered")
st.markdown(ui.CSS, unsafe_allow_html=True)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.keras")


def model_problem():
    if not os.path.exists(MODEL_PATH):
        return "model.keras was not found in the repository. Files found: " + ", ".join(sorted(os.listdir(HERE)))
    size = os.path.getsize(MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        head = f.read(4)
    if head != b"PK\x03\x04":
        return f"model.keras is not a valid Keras file (size {size} bytes). Please upload it again."
    return None


@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH, compile=False)


st.markdown(ui.hero_html(), unsafe_allow_html=True)
tab_predict, tab_about = st.tabs(["Predict", "About the project"])

with tab_predict:
    st.markdown(ui.note_html(), unsafe_allow_html=True)
    file = st.file_uploader("Upload a fundus / eye image", type=["jpg", "jpeg", "png"])

    if file is None:
        st.markdown(ui.empty_html(), unsafe_allow_html=True)
    else:
        img = Image.open(file).convert("RGB")
        left, right = st.columns(2, gap="large")
        with left:
            preview = img.copy()
            preview.thumbnail((520, 520))
            st.image(preview, caption="Uploaded image")
        with right:
            problem = model_problem()
            if problem:
                st.error(problem)
                st.stop()
            arr = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), axis=0)  # raw 0-255, model rescales inside
            try:
                with st.spinner("Predicting..."):
                    probs = get_model().predict(arr, verbose=0)[0]
            except Exception as e:
                st.error(f"Could not load or run the model: {type(e).__name__}: {e}")
                st.stop()
            st.markdown(ui.result_html(probs), unsafe_allow_html=True)

with tab_about:
    st.markdown(ui.about_html(), unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    if os.path.exists(os.path.join(HERE, "assets", "confusion_matrix.png")):
        c1.image(os.path.join(HERE, "assets", "confusion_matrix.png"), caption="Confusion matrix (test set)")
    if os.path.exists(os.path.join(HERE, "assets", "gradcam.png")):
        c2.image(os.path.join(HERE, "assets", "gradcam.png"), caption="Grad-CAM examples")
    st.markdown(ui.team_html(), unsafe_allow_html=True)
