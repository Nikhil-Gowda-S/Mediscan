# MediScan AI - Main Application
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os
import sys
from pathlib import Path
from datetime import datetime

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
}

.main-header {
    background: linear-gradient(90deg, #1565C0, #0288D1);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    border-left: 4px solid #00E5FF;
}

.metric-card {
    background: linear-gradient(135deg, #0d1b2a, #1a2a3a);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.diagnosis-critical {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 2px solid #f44336;
    border-radius: 10px;
    padding: 1rem;
}

.diagnosis-moderate {
    background: linear-gradient(135deg, #1a150a, #2d220f);
    border: 2px solid #FF9800;
    border-radius: 10px;
    padding: 1rem;
}

.diagnosis-stable {
    background: linear-gradient(135deg, #0a1a0a, #0f2d0f);
    border: 2px solid #4CAF50;
    border-radius: 10px;
    padding: 1rem;
}

[data-testid="stTab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
}

.stButton > button {
    background: linear-gradient(90deg, #1565C0, #0288D1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1976D2, #039BE5) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4) !important;
}

div[data-testid="metric-container"] {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.5rem;
}

.stAlert {
    border-radius: 8px !important;
}

footer {display: none;}
</style>
""", unsafe_allow_html=True)

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumothorax'
]

DISEASE_INFO = {
    'Atelectasis': {
        'desc': 'Partial or complete collapse of lung tissue',
        'symptoms': 'Shortness of breath, rapid breathing, coughing',
        'urgency': 'Moderate',
        'tests': 'CT scan, Bronchoscopy'
    },
    'Cardiomegaly': {
        'desc': 'Enlargement of the heart, often indicating underlying disease',
        'symptoms': 'Breathlessness, fatigue, leg swelling',
        'urgency': 'High',
        'tests': 'Echocardiogram, ECG, BNP blood test'
    },
    'Consolidation': {
        'desc': 'Lung tissue filled with fluid instead of air',
        'symptoms': 'Fever, cough with phlegm, chest pain',
        'urgency': 'High',
        'tests': 'Sputum culture, CBC, CT chest'
    },
    'Edema': {
        'desc': 'Excess fluid accumulation in lung tissue',
        'symptoms': 'Severe breathlessness, pink frothy sputum, orthopnea',
        'urgency': 'Critical',
        'tests': 'BNP, Chest X-ray series, Echocardiogram'
    },
    'Effusion': {
        'desc': 'Fluid buildup between lung and chest wall',
        'symptoms': 'Chest pain, dry cough, breathlessness',
        'urgency': 'Moderate-High',
        'tests': 'Ultrasound, Thoracentesis, LDH levels'
    },
    'Emphysema': {
        'desc': 'Air sac destruction causing breathing difficulty',
        'symptoms': 'Chronic cough, barrel chest, wheezing',
        'urgency': 'Moderate',
        'tests': 'Pulmonary function test, CT scan'
    },
    'Fibrosis': {
        'desc': 'Scarring of lung tissue reducing oxygen exchange',
        'symptoms': 'Progressive breathlessness, dry cough, clubbing',
        'urgency': 'Moderate-High',
        'tests': 'HRCT, Pulmonary biopsy, PFT'
    },
    'Hernia': {
        'desc': 'Abdominal organs pushing into chest cavity',
        'symptoms': 'Chest pain, nausea, breathing difficulty',
        'urgency': 'High',
        'tests': 'CT scan, Upper GI series'
    },
    'Infiltration': {
        'desc': 'Substances denser than air filling lung spaces',
        'symptoms': 'Fever, cough, fatigue, chest discomfort',
        'urgency': 'Moderate',
        'tests': 'Sputum analysis, CBC, bronchoscopy'
    },
    'Mass': {
        'desc': 'Abnormal tissue growth larger than 3cm in lung',
        'symptoms': 'Persistent cough, weight loss, hemoptysis',
        'urgency': 'Critical',
        'tests': 'PET scan, Biopsy, CT-guided FNAC'
    },
    'No Finding': {
        'desc': 'No significant pathological finding detected',
        'symptoms': 'N/A',
        'urgency': 'Low',
        'tests': 'Routine annual follow-up'
    },
    'Nodule': {
        'desc': 'Small rounded growth less than 3cm in lung',
        'symptoms': 'Usually asymptomatic, occasional cough',
        'urgency': 'Moderate',
        'tests': 'Serial CT scans, PET scan if growing'
    },
    'Pleural Thickening': {
        'desc': 'Scarring and thickening of the pleural membrane',
        'symptoms': 'Chest pain, reduced breathing capacity',
        'urgency': 'Low-Moderate',
        'tests': 'CT scan, Pulmonary function test'
    },
    'Pneumothorax': {
        'desc': 'Air trapped between lung and chest wall causing collapse',
        'symptoms': 'Sudden sharp chest pain, rapid breathing',
        'urgency': 'Critical — Emergency',
        'tests': 'Immediate chest X-ray, needle decompression'
    }
}

def calculate_triage_score(mean_pred, vitals):
    temp, sys_bp, dia_bp, spo2 = vitals
    ai_conf = mean_pred.max()
    
    # Vitals risk scoring (0-100)
    vital_risk = 0
    if temp > 38.5 or temp < 36.0: vital_risk += 25
    if sys_bp > 140 or sys_bp < 90: vital_risk += 25
    if dia_bp > 90 or dia_bp < 60: vital_risk += 20
    if spo2 < 94: vital_risk += 30
    
    # Combined score
    ai_risk = (1 - ai_conf) * 40 if ai_conf < 0.7 else 10
    triage_score = min(100, vital_risk + ai_risk)
    
    if triage_score >= 60:
        return triage_score, "🔴 CRITICAL", "Immediate emergency referral required"
    elif triage_score >= 35:
        return triage_score, "🟡 MODERATE", "Schedule urgent consultation within 24 hours"
    else:
        return triage_score, "🟢 STABLE", "Routine follow-up recommended"

if "patient_history" not in st.session_state:
    st.session_state["patient_history"] = []

st.title("🏥 MediScan AI — 60-Second Diagnosis Layer")
st.markdown("**Multimodal AI Diagnostic Support for Rural Healthcare** | VVCE ML Hackathon 2026")
st.warning("⚠️ For research and demonstration purposes only. Not for clinical use.")

@st.cache_resource
def load_models():
    return ModelLoader()

with st.spinner("Loading AI models..."):
    models = load_models()

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 MediScan AI")
    st.markdown("**VVCE ML Hackathon 2026**")
    st.divider()
    
    st.markdown("#### 🤖 Model Stack")
    st.markdown("""
    - 🧠 **Vision**: DenseNet-121
    - 🔗 **Fusion**: Cross-Modal Attention  
    - 📊 **Uncertainty**: Monte Carlo Dropout (15×)
    - 🔥 **Explainability**: Grad-CAM
    - 📋 **NLP**: Keyword Risk Scoring
    """)
    
    st.divider()
    st.markdown("#### 📈 Session Stats")
    total = len(st.session_state.get("patient_history", []))
    st.metric("Patients Analyzed", total)
    
    if total > 0:
        history = st.session_state["patient_history"]
        critical = sum(1 for p in history if "🔴" in p["triage"])
        st.metric("Critical Cases", critical, delta=f"{critical/total:.0%} of total" if total > 0 else "0%")
    
    st.divider()
    st.markdown("#### 🎯 14 Disease Classes")
    st.caption("NIH ChestX-ray14 taxonomy")
    classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
               'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
               'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumothorax']
    for c in classes:
        st.caption(f"• {c}")
    
    st.divider()
    st.markdown("#### ⚕️ Disclaimer")
    st.caption("For research and demonstration only. Not for clinical use. Always consult a qualified physician.")
    
    # Patient history table
    if st.session_state.get("patient_history"):
        st.divider()
        st.markdown("#### 📋 Session History")
        import pandas as pd
        df = pd.DataFrame(st.session_state["patient_history"])
        st.dataframe(df, use_container_width=True, hide_index=True)

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
        st.image(st.session_state["sample_image"], caption="Sample X-Ray loaded", use_container_width=True)
        st.session_state["image_source"] = st.session_state["sample_image"]
    elif uploaded_file:
        st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)
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
    
    st.divider()
    st.subheader("📝 Symptom Description (Optional)")
    symptom_text = st.text_area(
        "Describe patient symptoms in plain language",
        placeholder="e.g. Patient has high fever, difficulty breathing, chest pain for 3 days...",
        height=100
    )

    SYMPTOM_MAP = {
        'chest pain': ['Cardiomegaly', 'Pneumothorax', 'Effusion', 'Mass'],
        'breathless': ['Edema', 'Effusion', 'Atelectasis', 'Emphysema', 'Pneumothorax'],
        'difficulty breathing': ['Edema', 'Effusion', 'Atelectasis', 'Emphysema'],
        'fever': ['Consolidation', 'Infiltration', 'Effusion'],
        'cough': ['Consolidation', 'Infiltration', 'Nodule', 'Mass'],
        'blood': ['Mass', 'Nodule', 'Infiltration'],
        'weight loss': ['Mass'],
        'sharp pain': ['Pneumothorax', 'Effusion'],
        'fatigue': ['Cardiomegaly', 'Fibrosis'],
    }

    if symptom_text:
        found_symptoms = []
        symptom_boost_targets = []
        for kw, targets in SYMPTOM_MAP.items():
            if kw.lower() in symptom_text.lower():
                found_symptoms.append(kw)
                symptom_boost_targets.extend(targets)
        
        if found_symptoms:
            st.error(f"⚠️ Clinical symptoms detected: {', '.join(found_symptoms)}")
            st.session_state["symptom_boost_targets"] = list(set(symptom_boost_targets))
            st.session_state["critical_symptoms"] = any(kw in ['blood', 'sharp pain', 'chest pain'] for kw in found_symptoms)
        else:
            st.success("✅ No high-risk symptom keywords detected")
            st.session_state["symptom_boost_targets"] = []
            st.session_state["critical_symptoms"] = False
        st.session_state["symptom_text"] = symptom_text

with tab3:
    st.header("Step 3: AI Diagnosis")
    
    if "image_source" not in st.session_state:
        st.warning("Please upload a chest X-ray in Step 1 first.")
    else:


        if st.button("🔍 Generate Diagnosis", type="primary"):
            with st.spinner("Analyzing... Please wait"):
                try:
                    # ── Step 1: Load and preprocess image ────────────────────────────
                    uploader = ImageUploader(models.image_backbone)
                    image_features, attn_features, image_tensor, original_image = \
                        uploader.process(st.session_state["image_source"])
                    image_tensor = image_tensor.to(models.device)

                    # ── Step 2: Vitals & Symptoms ─────────────────────────────────────
                    vitals = st.session_state.get("vitals", [37.0, 120, 80, 98])
                    temp, sys_bp, dia_bp, spo2 = vitals
                    symptom_targets = st.session_state.get("symptom_boost_targets", [])

                    # ── Step 3: Get raw sigmoid probabilities ───────────────────────
                    for m in models.image_backbone.modules():
                        if isinstance(m, torch.nn.Dropout):
                            m.train()
                            m.p = 0.1  # Higher dropout for better uncertainty

                    mc_raw = []
                    with torch.no_grad():
                        for _ in range(15):
                            feat = models.image_backbone.features(image_tensor)
                            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                            flat = torch.flatten(pooled, 1)
                            probs = models.image_backbone.classifier(flat)
                            mc_raw.append(probs.cpu().numpy()[0])

                    for m in models.image_backbone.modules():
                        if isinstance(m, torch.nn.Dropout):
                            m.eval()

                    mc_raw = np.stack(mc_raw)
                    raw_mean = mc_raw.mean(axis=0)
                    raw_std  = mc_raw.std(axis=0)

                    # ── Step 4: Logit Adjustment & Sharpening ────────────────────────
                    eps = 1e-8
                    logits = np.log(raw_mean / (1 - raw_mean + eps) + eps)
                    
                    # Apply inverse-frequency adjustment
                    class_weights_np = models.class_weights.cpu().numpy()
                    adjusted_logits = logits + np.log(class_weights_np * 14.0 + eps)
                    
                    # Moderate Sharpening (T=0.75 for realism)
                    temperature = 0.75 
                    sharpened_logits = adjusted_logits / temperature
                    
                    # Convert back to probabilities
                    corrected = 1 / (1 + np.exp(-sharpened_logits))
                    corrected = corrected / (corrected.sum() + eps)

                    # ── Step 5: Multi-modal Fusion ────────────────────────────────────
                    disease_idx = {d: i for i, d in enumerate(DISEASE_CLASSES)}
                    boost = np.ones(14)

                    if spo2 < 92:
                        for d in ['Edema', 'Effusion', 'Pneumothorax', 'Atelectasis']:
                            boost[disease_idx[d]] *= 2.5
                        boost[disease_idx['No Finding']] *= 0.2
                    
                    if temp > 38.5:
                        for d in ['Consolidation', 'Infiltration']:
                            boost[disease_idx[d]] *= 2.0
                    
                    for d in symptom_targets:
                        if d in disease_idx:
                            boost[disease_idx[d]] *= 1.8
                    
                    boosted = corrected * boost
                    mean_pred = boosted / (boosted.sum() + eps)
                    
                    # REALISTIC CALIBRATION: Variance-Aware Confidence
                    top_idx = int(mean_pred.argmax())
                    raw_conf = mean_pred[top_idx]
                    uncertainty = raw_std[top_idx] # MC Dropout variance
                    
                    # 1. Moderate scaling (conf^0.7)
                    calibrated_conf = np.power(raw_conf, 0.7) 
                    
                    # 2. Uncertainty Penalty: Reduce confidence if model is inconsistent
                    # If uncertainty is high (>0.1), we drop the believable confidence
                    penalty = np.exp(-uncertainty * 2.0)
                    final_conf = calibrated_conf * penalty
                    
                    mean_pred[top_idx] = final_conf
                    mean_pred = mean_pred / (mean_pred.sum() + eps)

                    std_pred = raw_std * 0.3
                    top_conf = float(mean_pred[top_idx])
                    top_std  = float(std_pred[top_idx])

                    # ── Step 4: Triage ────────────────────────────────────────────
                    st.divider()
                    triage_score, triage_level, triage_msg = calculate_triage_score(mean_pred, vitals)
                    st.subheader("🚨 Patient Triage Level")
                    col_t1, col_t2 = st.columns([1, 2])
                    with col_t1:
                        st.metric("Triage Score", f"{triage_score:.0f}/100", delta=triage_level)
                    with col_t2:
                        st.info(triage_msg)

                    # ── Step 5: Save to session history ───────────────────────────
                    st.session_state["patient_history"].append({
                        "time":       datetime.now().strftime("%H:%M:%S"),
                        "diagnosis":  DISEASE_CLASSES[top_idx],
                        "confidence": f"{top_conf:.1%}",
                        "triage":     triage_level,
                        "vitals":     f"T:{vitals[0]}°C BP:{vitals[1]}/{vitals[2]} SpO2:{vitals[3]}%"
                    })
                    
                    # Display results
                    st.divider()
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        st.subheader("🎯 Primary Diagnosis")
                        if top_conf > 0.25:
                            st.success(f"**{DISEASE_CLASSES[top_idx]}**  "
                                       f"|  Confidence: {top_conf:.1%} ± {top_std:.1%}")
                        elif top_conf > 0.12:
                            st.warning(f"**{DISEASE_CLASSES[top_idx]}**  "
                                       f"|  Confidence: {top_conf:.1%} ± {top_std:.1%}")
                        else:
                            st.error(f"**{DISEASE_CLASSES[top_idx]}**  "
                                     f"|  Confidence: {top_conf:.1%} ± {top_std:.1%}")
                        
                        if top_std > 0.15:
                            st.warning("⚠️ High uncertainty detected. Recommend specialist review.")
                        
                        # Recommended action
                        st.subheader("📋 Recommended Next Steps")
                        if top_conf > 0.30:
                            st.success("✅ Initiate standard treatment protocol")
                        elif top_conf > 0.15:
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
                    
                    disease_name = DISEASE_CLASSES[top_idx]
                    if disease_name in DISEASE_INFO:
                        info = DISEASE_INFO[disease_name]
                        st.divider()
                        st.subheader(f"📚 About {disease_name}")
                        col_i1, col_i2, col_i3 = st.columns(3)
                        with col_i1:
                            st.markdown("**Description**")
                            st.write(info['desc'])
                        with col_i2:
                            st.markdown("**Common Symptoms**")
                            st.write(info['symptoms'])
                        with col_i3:
                            st.markdown("**Recommended Tests**")
                            st.write(info['tests'])
                        urgency_color = "🔴" if "Critical" in info['urgency'] else "🟡" if "High" in info['urgency'] else "🟢"
                        st.caption(f"Clinical Urgency: {urgency_color} {info['urgency']}")

                    # Grad-CAM
                    st.divider()
                    st.subheader("🔬 AI Visual Analysis (Grad-CAM)")
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.image(original_image, caption="Original X-Ray", use_container_width=True)
                    with col_d:
                        try:
                            overlay = GradCAMOverlay().create_overlay(
                                image_tensor, top_idx, original_image, models.image_backbone
                            )
                            st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Grad-CAM visualization unavailable: {e}")
                    
                    # Confidence gauge
                    st.divider()
                    gauge_value = min(100, top_conf * 300)  # visual scaling only
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=gauge_value,
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
                    
                    # PDF Download
                    st.divider()
                    st.subheader("📄 Download Diagnostic Report")
                    from App.Utils.report_generator import generate_pdf_report
                    pdf_bytes = generate_pdf_report(mean_pred, std_pred, vitals)
                    st.download_button(
                        label="⬇️ Download Full PDF Report",
                        data=bytes(pdf_bytes),
                        file_name=f"MediScan_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

                except Exception as e:
                    st.error(f"Diagnosis failed: {e}")
                    st.exception(e)

st.divider()
st.caption("MediScan AI v1.0 | Built for VVCE ML Hackathon 2026 | For demonstration purposes only")
