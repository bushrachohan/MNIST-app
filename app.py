"""
app.py — MNIST Digit Recogniser
Deploy at: https://share.streamlit.io
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.set_page_config(page_title="MNIST Digit Recogniser", page_icon="✍️")

st.title("✍️ MNIST Digit Recogniser")
st.write("Upload a handwritten digit image or pick a sample from the test set.")

@st.cache_resource
def get_model():
    return load_model("mnist_model.keras")

@st.cache_resource
def get_test_data():
    from tensorflow.keras.datasets import mnist
    (_, _), (X_test, y_test) = mnist.load_data()
    return X_test, y_test

# ── Input ─────────────────────────────────────────────────────────────────────
method = st.radio("Input method", ["Upload image", "Use MNIST sample"], horizontal=True)

img_array = None

if method == "Upload image":
    uploaded = st.file_uploader("Upload a digit image (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("L")
        img = ImageOps.invert(img)                  # white digit on black bg
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        st.image(img, caption="Your image (28×28)", width=140)

else:
    X_test, y_test = get_test_data()
    idx = st.number_input("MNIST test-set index (0 – 9999)", min_value=0, max_value=9999, value=0, step=1)
    img_array = X_test[int(idx)] / 255.0
    st.image(
        (img_array * 255).astype(np.uint8),
        caption=f"Sample #{idx}  |  True label: {y_test[int(idx)]}",
        width=140,
    )

# ── Predict ───────────────────────────────────────────────────────────────────
if img_array is not None and st.button("🔢 Predict", use_container_width=True):
    model = get_model()
    probs = model.predict(img_array.reshape(1, 28, 28), verbose=0)[0]
    pred  = int(np.argmax(probs))

    st.success(f"**Predicted digit: {pred}**  —  confidence {probs[pred]*100:.1f}%")

    st.subheader("All class probabilities")
    cols = st.columns(10)
    for i, (col, p) in enumerate(zip(cols, probs)):
        col.metric(label=str(i), value=f"{p*100:.0f}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("ANN: Flatten → 128 → 60 → 10 · Adam · 4 epochs  |  Code With Leo 🦁")
