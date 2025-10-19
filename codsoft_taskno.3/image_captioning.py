import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Set Streamlit page config
st.set_page_config(page_title="🧠 BLIP Image Captioning", layout="centered")
st.title("📸 Image Captioning ")

# Load BLIP model only once
@st.cache_resource
def load_blip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return model, processor, device

model, processor, device = load_blip_model()

# File uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    raw_image = Image.open(uploaded).convert("RGB")
    st.image(raw_image, caption="Your Image", use_container_width=True)

    with st.spinner("Generating caption with BLIP... 🧠"):
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.success(f"📝 Caption: {caption}")