import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import load_custom_cnn, load_resnet18

st.set_page_config(
    page_title="Ant vs Bee Classifier",
    page_icon="🐝",
    layout="centered"
)

CLASS_NAMES = ['Ant 🐜', 'Bee 🐝']
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    custom = load_custom_cnn("custom_cnn_model.pth", DEVICE)
    resnet = load_resnet18("resnet18_model.pth",     DEVICE)
    return custom, resnet

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def predict(model, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs   = F.softmax(outputs, dim=1)[0]
        pred    = torch.argmax(probs).item()
    return pred, probs.cpu().numpy()

st.title("🐜 Ant vs 🐝 Bee Classifier")
st.markdown("Upload an image and both models will classify it!")
st.divider()

custom_model, resnet_model = load_models()

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.divider()

    tensor = preprocess(image)

    custom_pred, custom_probs = predict(custom_model, tensor)
    resnet_pred, resnet_probs = predict(resnet_model, tensor)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Custom CNN")
        st.metric(label="Prediction", value=CLASS_NAMES[custom_pred])
        st.markdown("**Confidence:**")
        for i, cls in enumerate(CLASS_NAMES):
            st.progress(
                float(custom_probs[i]),
                text=f"{cls}: {custom_probs[i]*100:.1f}%"
            )

    with col2:
        st.subheader("🏆 ResNet18")
        st.metric(label="Prediction", value=CLASS_NAMES[resnet_pred])
        st.markdown("**Confidence:**")
        for i, cls in enumerate(CLASS_NAMES):
            st.progress(
                float(resnet_probs[i]),
                text=f"{cls}: {resnet_probs[i]*100:.1f}%"
            )

    st.divider()

    if custom_pred == resnet_pred:
        st.success(f"✅ Both models agree: **{CLASS_NAMES[custom_pred]}**")
    else:
        st.warning("⚠️ Models disagree! ResNet18 is generally more reliable.")

    st.markdown("### Confidence Comparison")
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(CLASS_NAMES))
    width = 0.3
    ax.bar(x - width/2, custom_probs, width, label='Custom CNN', color='steelblue', alpha=0.85)
    ax.bar(x + width/2, resnet_probs, width, label='ResNet18',   color='tomato',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel('Confidence')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)