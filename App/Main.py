# MediScan AI - Main Application
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from App.Utils.Model_Loader import ModelLoader
from App.Components.Image_Uploader import ImageUploader
from App.Components.Results_Display import ResultDisplay
from App.Components.Explainability import GradCAMOverlay

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🏥",
    layout="wide"
)

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumothorax'
]

st.title("🏥 MediScan AI — 60-Second Diagnosis Layer")
st.markdown("**Multimodal AI Diagnostic Support for Rural Healthcare** | VCE ML Hackathon 2026")
st.warning("⚠️ For research and demonstration purposes only. Not for clinical use.")

@st.cache_resource
def load_models():
    return ModelLoader()

with st.spinner("Loading AI models..."):
    models = load_models()

# Sidebar
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.info("Image Encoder: DenseNet-121\nClassifier: Cross-Modal Fusion\nExplainability: Grad-CAM\nUncertainty: Monte Carlo Dropout")
    st.divider()
    st.header("📂 Sample Images")
    sample_dir = "Data/Samples"
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg','.png','.jpeg'))]
        if samples:
            selected = st.selectbox("Load a sample X-ray", samples)
            if st.button("Load Sample"):
                st.session_state["sample_image"] = os.path.join(sample_dir, selected)

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload X-Ray", "🩺 Enter Vitals", "📊 Results"])

with tab1:
    st.header("Step 1: Upload Chest X-Ray")
    st.info("Upload a grayscale chest X-ray image (JPG or PNG)")
    
    uploaded_file = st.file_uploader("Choose X-Ray image", type=["jpg","png","jpeg"])
    
    if "sample_image" in st.session_state and not uploaded_file:
        st.image(st.session_state["sample_image"], caption="Sample X-Ray loaded", use_column_width=True)
        st.session_state["image_source"] = st.session_state["sample_image"]
    elif uploaded_file:
        st.image(uploaded_file, caption="Uploaded X-Ray", use_column_width=True)
        st.session_state["image_source"] = uploaded_file
        st.success("✅ X-Ray uploaded successfully")

with tab2:
    st.header("Step 2: Enter Patient Vitals")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("🌡️ Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
        sys_bp = st.number_input("💉 Systolic BP (mmHg)", min_value=80, max_value=200, value=120)
    with col2:
        dia_bp = st.number_input("💉 Diastolic BP (mmHg)", min_value=40, max_value=140, value=80)
        spo2 = st.number_input("🫁 SpO2 (%)", min_value=70, max_value=100, value=98)
    
    st.session_state["vitals"] = [temp, sys_bp, dia_bp, spo2]
    st.success("✅ Vitals recorded")

with tab3:
    st.header("Step 3: AI Diagnosis")
    
    if "image_source" not in st.session_state:
        st.warning("Please upload a chest X-ray in Step 1 first.")
    else:
        if st.button("🔍 Generate Diagnosis", type="primary"):
            with st.spinner("Analyzing... Please wait"):
                try:
                    # Load and process image
                    uploader = ImageUploader(models.image_backbone)
                    image_features, attn_features, image_tensor, original_image = uploader.process(
                        st.session_state["image_source"]
                    )
                    
                    # Vitals tensor
                    vitals = st.session_state.get("vitals", [37.0, 120, 80, 98])
                    vital_features = torch.tensor([vitals]).float().to(models.device)
                    
                    # Audio is removed — use zero vector
                    audio_features = torch.zeros(1, 256).to(models.device)
                    
                    # Monte Carlo Dropout — 15 forward passes
                    models.fusion_classifier.train()  # enable dropout
                    predictions = []
                    for _ in range(15):
                        with torch.no_grad():
                            pred = models.fusion_classifier(attn_features, audio_features, vital_features)
                        predictions.append(pred.cpu().numpy()[0])
                    models.fusion_classifier.eval()
                    
                    predictions = np.stack(predictions)
                    mean_pred = predictions.mean(axis=0)
                    std_pred = predictions.std(axis=0)
                    
                    top_idx = mean_pred.argmax()
                    top_conf = mean_pred[top_idx]
                    top_std = std_pred[top_idx]
                    
                    # Display results
                    st.divider()
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        st.subheader("🎯 Primary Diagnosis")
                        if top_conf > 0.7:
                            st.success(f"**{DISEASE_CLASSES[top_idx]}**\nConfidence: {top_conf:.1%} ± {top_std:.1%}")
                        elif top_conf > 0.5:
                            st.warning(f"**{DISEASE_CLASSES[top_idx]}**\nConfidence: {top_conf:.1%} ± {top_std:.1%}")
                        else:
                            st.error(f"**{DISEASE_CLASSES[top_idx]}**\nConfidence: {top_conf:.1%} ± {top_std:.1%}")
                        
                        if top_std > 0.15:
                            st.warning("⚠️ High uncertainty detected. Recommend specialist review.")
                        
                        # Recommended action
                        st.subheader("📋 Recommended Next Steps")
                        if top_conf > 0.85:
                            st.success("✅ Initiate standard treatment protocol")
                        elif top_conf > 0.65:
                            st.info("🔬 Order confirmatory diagnostic tests")
                        else:
                            st.error("🚨 Urgent specialist referral required")
                    
                    with col_b:
                        st.subheader("📊 Differential Diagnosis")
                        top5_idx = mean_pred.argsort()[-5:][::-1]
                        fig = go.Figure(go.Bar(
                            x=[mean_pred[i] * 100 for i in top5_idx],
                            y=[DISEASE_CLASSES[i] for i in top5_idx],
                            orientation='h',
                            marker_color=['#2196F3' if i == top_idx else '#90CAF9' for i in top5_idx]
                        ))
                        fig.update_layout(
                            xaxis_title="Confidence (%)",
                            height=300,
                            margin=dict(l=0, r=0, t=20, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Grad-CAM
                    st.divider()
                    st.subheader("🔬 AI Visual Analysis (Grad-CAM)")
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.image(original_image, caption="Original X-Ray", use_column_width=True)
                    with col_d:
                        try:
                            overlay = GradCAMOverlay().create_overlay(
                                image_tensor, top_idx, original_image, models.image_backbone
                            )
                            st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Grad-CAM visualization unavailable: {e}")
                    
                    # Confidence gauge
                    st.divider()
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=top_conf * 100,
                        title={'text': "Overall Confidence (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#2196F3"},
                            'steps': [
                                {'range': [0, 50], 'color': "#FFCDD2"},
                                {'range': [50, 70], 'color': "#FFF9C4"},
                                {'range': [70, 100], 'color': "#C8E6C9"}
                            ]
                        }
                    ))
                    gauge.update_layout(height=300)
                    st.plotly_chart(gauge, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Diagnosis failed: {e}")
                    st.exception(e)

st.divider()
st.caption("MediScan AI v1.0 | Built for VCE ML Hackathon 2026 | For demonstration purposes only")
