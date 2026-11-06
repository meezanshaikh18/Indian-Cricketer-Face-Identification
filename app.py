"""Streamlit demo:  streamlit run app.py"""
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import config
from src.inference import load_bundle, predict

st.set_page_config(page_title="Cricketer Face Identification", page_icon="🏏", layout="centered")


@st.cache_resource(show_spinner="Loading face models (first run downloads them)...")
def get_resources():
    from src.face_embedder import FaceEmbedder
    return FaceEmbedder(), load_bundle()


st.title("🏏 Indian Cricketer Face Identification")
st.caption("Upload a photo and the model tells you which player it is, using FaceNet face embeddings "
           "and a trained classifier.")

try:
    embedder, bundle = get_resources()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence needed (otherwise 'Unknown')", 0.0, 0.95, float(bundle["threshold"]), 0.05)
    st.markdown(f"**Model:** {bundle['model_name']}")
    st.markdown("**Players the model knows:**")
    for name in bundle["classes"]:
        st.write(f"- {name}")
    st.caption("Educational project. Only works for the players above; anyone else should come out as Unknown.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    result = predict(image, embedder, bundle, threshold)
    left, right = st.columns(2)
    left.image(image, caption="Your image", use_container_width=True)
    if result["status"] == "no_face":
        right.warning("No clear face found. Try a sharper, front-facing photo.")
    else:
        right.image(result["face"], caption="Detected face", use_container_width=True)
        if result["unknown"]:
            st.warning(f"**Unknown player** - closest was {result['best_guess']} "
                       f"({result['confidence']:.0%}), below the {threshold:.0%} threshold.")
        else:
            st.success(f"**{result['label']}** - confidence {result['confidence']:.0%}")
        chart = pd.DataFrame(result["top"], columns=["Player", "Probability"]).set_index("Player")
        st.bar_chart(chart)
