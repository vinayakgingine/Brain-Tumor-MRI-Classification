
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Brain MRI Classifier", page_icon="🧠")
st.title("Brain MRI Image Classification")
st.warning(
    "Educational demonstration only. This tool is not a medical device "
    "and cannot diagnose or rule out a brain tumor. Consult a qualified clinician."
)

MODEL_PATH = "Models/brain_tumor_efficientnet.keras"
LABEL_PATH = "Models/class_names.json"

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run educational classification demo"):
        model, labels = load_artifacts()
        arr = np.asarray(image, dtype=np.float32)[None, ...]
        scores = model.predict(arr, verbose=0)[0]
        ranking = np.argsort(scores)[::-1]

        st.subheader("Model output")
        st.write("Top dataset label:", labels[int(ranking[0])])
        st.caption("Scores are not calibrated clinical probabilities.")
        for idx in ranking:
            st.write(f"{labels[idx]}: {scores[idx]*100:.2f}%")
