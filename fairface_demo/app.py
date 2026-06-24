import os
import sys
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
from model_def import FairFaceModel, NUM_CLASSES

# CONFIG
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RACE_LABELS = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino_Hispanic"
]

GENDER_LABELS = [
    "Male",
    "Female"
]

AGE_LABELS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70"
]

# PAGE

st.set_page_config(
    page_title="FairFace AI Analyzer",
    layout="wide"
)

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #fff7fb;
}

.block-container {
    padding-top: 2rem;
}

.result-card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    border: 2px solid #ffd6e7;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(255,192,203,0.15);
}

.result-title {
    color: #d63384;
    font-size: 18px;
    font-weight: 600;
}

.result-value {
    color: #212529;
    font-size: 28px;
    font-weight: 700;
}

.result-confidence {
    color: #666;
    font-size: 14px;
}

h1 {
    color: #d63384;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

st.title("FairFace AI Analyzer")

st.markdown(
    """
    <center>
    Dự đoán Race • Gender • Age bằng Vision Transformer
    </center>
    """,
    unsafe_allow_html=True
)

# DEVICE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# SIDEBAR

st.sidebar.header("Environment")

st.sidebar.write("Python")
st.sidebar.code(sys.executable)

st.sidebar.write("Torch")
st.sidebar.code(torch.__version__)

st.sidebar.write("TIMM")
st.sidebar.code(timm.__version__)

st.sidebar.write("Device")
st.sidebar.code(str(device))

# MODEL LOADER
@st.cache_resource
def load_model(checkpoint_path):
    model = FairFaceModel(
        NUM_CLASSES,
        timm_id="vit_small_patch16_224",
        dropout_rate=0.4,
        name="ViT-Small"
    )

    debug_info = []

    if not os.path.exists(checkpoint_path):
        debug_info.append(f"Checkpoint NOT FOUND: {checkpoint_path}")
        return None, debug_info
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        debug_info.append("Checkpoint loaded successfully.")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        debug_info.append(f"Missing layers: {len(missing)}")
        debug_info.append(f"Unexpected layers: {len(unexpected)}")

        model.to(device)
        model.eval()

        debug_info.append("Model loaded and ready.")

        return model, debug_info

    except Exception as e:
        debug_info.append(f"ERROR: {str(e)}")
        return None, debug_info

# PREPROCESS
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])
    return transform(image).unsqueeze(0)

# LOAD MODEL
checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path",
    "ViT_Small_best.pth"
)
model, debug_info = load_model(checkpoint_path)

# DEBUG
with st.expander("Debug Information", expanded=False):
    for item in debug_info:
        st.write(item)

# STATUS
if model is None:
    st.error("Model failed to load.")
else:
    st.success("Model loaded successfully.")

# UPLOAD
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=300)

    with col2:
        st.subheader("Prediction Result")
        try:
            input_tensor = preprocess_image(image).to(device)
            with torch.no_grad():
                race_out, gender_out, age_out = model(input_tensor)
            race_probs = torch.softmax(race_out, dim=1)[0]
            gender_probs = torch.softmax(gender_out, dim=1)[0]
            age_probs = torch.softmax(age_out, dim=1)[0]

            race_pred = torch.argmax(race_probs).item()
            gender_pred = torch.argmax(gender_probs).item()
            age_pred = torch.argmax(age_probs).item()

            race_conf = race_probs.max().item() * 100
            gender_conf = gender_probs.max().item() * 100
            age_conf = age_probs.max().item() * 100

            st.markdown(f'''
            <div class="result-card">
                <div class="result-title">🌎 Race</div>
                <div class="result-value">{RACE_LABELS[race_pred]}</div>
                <div class="result-confidence">
                    Confidence: {race_conf:.2f}%
                </div>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown(f'''
            <div class="result-card">
                <div class="result-title">👤 Gender</div>
                <div class="result-value">{GENDER_LABELS[gender_pred]}</div>
                <div class="result-confidence">
                    Confidence: {gender_conf:.2f}%
                </div>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown(f'''
            <div class="result-card">
                <div class="result-title">🎂 Age Group</div>
                <div class="result-value">{AGE_LABELS[age_pred]}</div>
                <div class="result-confidence">
                    Confidence: {age_conf:.2f}%
                </div>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Top 3 Race Predictions")
            top_probs, top_idx = torch.topk(race_probs, 3)

            for prob, idx in zip(top_probs, top_idx):
                st.write(f"**{RACE_LABELS[idx]}** — {prob.item()*100:.2f}%")
                st.progress(float(prob.item()))

            st.success("Analysis completed successfully!")

        except Exception as e:
            st.error(f"Inference Error: {str(e)}")