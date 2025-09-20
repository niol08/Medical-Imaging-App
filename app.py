
import streamlit as st
from src.services.inference import run_inference

st.set_page_config(page_title="Medical Imaging Classifier", layout="wide")

st.title("Medical Imaging Disease Classifier")
st.markdown(
    "Upload a **CT** (DICOM) or **Chest X-ray** (JPG/PNG) scan to get an AI-based classification "
    "and clinician-friendly insights."
)


st.sidebar.header("Settings")
modality = st.sidebar.radio("Select Imaging Modality", ["CT", "X-ray"])


uploaded_file = st.file_uploader(
    f"Upload {modality} image",
    type=["jpg", "jpeg", "png", "dcm", "dicom"]
)

if uploaded_file is not None:
    import tempfile, os
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner("Running AI inference..."):
        result = run_inference(modality, tmp_path)

    st.subheader("Prediction Results")
    st.write(f"**Predicted Condition:** {result['label_name']}")
    st.write(f"**Confidence:** {result['confidence']:.2%}")

    st.subheader("AI Insight")
    st.info(result["ai_insight"])

    ext = uploaded_file.name.lower().split(".")[-1]
    if ext in ["jpg", "jpeg", "png"]:
        from PIL import Image
        img = Image.open(tmp_path)
        st.image(img, caption=f"Uploaded {modality} Image", use_container_width=True)

    elif ext in ["dcm", "dicom"]:
        import pydicom
        import numpy as np
        from PIL import Image

        ds = pydicom.dcmread(tmp_path)
        arr = ds.pixel_array.astype(float)

        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
        arr = arr.astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB")
        st.image(img, caption=f"Uploaded {modality} DICOM", use_container_width=True)
