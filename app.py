import streamlit as st

import asyncio

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from models.multitask_model import FishSegmentationClassificationModel
from scipy.ndimage import gaussian_filter
import gc

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Set Streamlit page config
st.set_page_config(page_title="Fish Segmentation & Classification", layout="wide")

# Load model with cache to optimize loading
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FishSegmentationClassificationModel(num_classes=23)
    model.load_state_dict(torch.load('checkpoints/best_model_epoch18.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class_labels = {
    0: "Dascyllus reticulatus",
    1: "Plectroglyphidodon dickii",
    2: "Chromis chrysura",
    3: "Amphiprion clarkii",
    4: "Chaetodon lunulatus",
    5: "Chaetodon trifascialis",
    6: "Myripristis kuntee",
    7: "Acanthurus nigrofuscus",
    8: "Hemigymnus fasciatus",
    9: "Neoniphon sammara",
    10: "Abudefduf vaigiensis",
    11: "Canthigaster valentini",
    12: "Pomacentrus moluccensis",
    13: "Zebrasoma scopas",
    14: "Hemigymnus melapterus",
    15: "Lutjanus fulvus",
    16: "Scolopsis bilineata",
    17: "Scaridae",
    18: "Pempheris vanicolensis",
    19: "Zanclus cornutus",
    20: "Neoglyphidodon nigroris",
    21: "Balistapus undulatus",
    22: "Siganus fuscescens"
}

def smooth_mask(mask_np, sigma=1.5):
    return gaussian_filter(mask_np.astype(float), sigma=sigma)

# Prediction function
def predict(image, sigma):
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        seg_output, cls_output = model(image)
    
    # Segmentation mask
    seg_output = torch.sigmoid(seg_output.squeeze(0)).cpu().numpy()
    seg_mask = (seg_output[0] > 0.5).astype(np.uint8)

    # Smooth the mask
    seg_mask = smooth_mask(seg_mask, sigma=sigma)
    seg_mask = (seg_mask > 0.5).astype(np.uint8)

    # Classification
    probs = F.softmax(cls_output, dim=1)
    confidence, predicted_class = torch.max(probs, 1)

    return seg_mask, predicted_class.item(), confidence.item()

# Create overlay function
def create_overlay(image_pil, mask_np, mask_color=(255, 0, 0)):
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(image_pil.size)
    
    # Create a colored mask based on user's color choice
    color_mask = Image.new("RGB", image_pil.size, mask_color)
    color_mask.putalpha(mask_pil)

    image_with_overlay = image_pil.convert("RGBA")
    combined = Image.alpha_composite(image_with_overlay, color_mask)

    return combined.convert("RGB")

# Streamlit app UI
st.title("🐟 Fish Segmentation & Classification")

uploaded_file = st.file_uploader("Upload an underwater image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Brightness and Contrast sliders
    st.sidebar.header("Image Adjustments")
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.05)

    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(brightness)

    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(contrast)

    st.image(image, caption='Adjusted Uploaded Image', use_container_width=True)

    # Mask transparency and color
    st.sidebar.header("Mask Settings")
    mask_color_hex = st.sidebar.color_picker("Mask Color", "#FF0000")
    mask_color = tuple(int(mask_color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    # Mask smoothing control
    sigma = st.sidebar.slider("Mask Smoothness (Gaussian Sigma)", 0.5, 5.0, 1.5, 0.1)

    if st.button("Predict"):
        with st.spinner('Predicting...'):
            mask, label, confidence = predict(image, sigma=sigma)

            st.subheader(f"Predicted Class: {class_labels[label]} (Confidence: {confidence:.2f})")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Segmentation Mask")
                st.image(mask * 255, caption='Predicted Mask', use_container_width=True, clamp=True)
            with col2:
                st.subheader("Overlayed Image")
                overlay_img = create_overlay(image, mask, mask_color=mask_color)
                st.image(overlay_img, caption='Image with Segmentation Overlay', use_container_width=True)

            gc.collect()
            torch.cuda.empty_cache()
