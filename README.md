# 🏥 MediScan AI — 60-Second Diagnosis Layer

**Multimodal AI Diagnostic Support for Rural Healthcare**  
*Built for the VVCE ML Hackathon 2026*

MediScan AI is an advanced medical decision support system designed to bridge the healthcare gap in resource-constrained environments. Unlike standard "black-box" AI models, MediScan fuses **Chest X-Ray analysis** with **Patient Vitals** and **Plain-Language Symptoms** to provide a realistic, explainable, and clinically grounded diagnosis.

---

## 🚀 Key Features

### 🧠 1. Multi-Modal Fusion Engine
MediScan doesn't rely solely on images. It integrates:
- **Vision**: Fine-tuned CheXNet (DenseNet-121) for 14 chest pathologies.
- **Vitals**: Real-time analysis of SpO2, Temperature, and BP.
- **NLP**: Symptom-to-pathology mapping from patient descriptions.

### 📊 2. Reliability & Realism
- **Monte Carlo Dropout**: 15x stochastic sampling to measure model uncertainty.
- **Logit Adjustment**: Mathematical debiasing to handle the 80:1 class imbalance in the NIH dataset.
- **Variance-Aware Confidence**: Confidence scores that drop when the model is confused, ensuring "believable" AI.

### 🔬 3. Trust through Explainability
- **Grad-CAM Localization**: High-resolution spatial heatmaps (norm5) that highlight pathology hotspots.
- **Clinical Triage**: Integrated emergency referral scoring (Critical/Moderate/Stable).

---

## 🛠️ Tech Stack
- **Framework**: Streamlit (Dashboard)
- **Deep Learning**: PyTorch, Torchvision
- **Explainability**: Grad-CAM (Custom Implementation)
- **Visualization**: Plotly, OpenCV
- **Reporting**: FPDF (PDF Diagnostics), python-pptx (Pitch Deck Generation)

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nikhil-Gowda-S/Mediscan.git
   cd Mediscan
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run App/Main.py
   ```

---

## 📂 Project Structure
- `App/Main.py`: Primary interface and inference orchestration.
- `App/Utils/Model_Loader.py`: Backbone initialization and weight loading.
- `App/Components/Explainability.py`: Grad-CAM implementation.
- `App/Components/Image_Uploader.py`: Aspect-ratio preserving preprocessing.
- `App/Models/`: Fine-tuned weights and architecture definitions.

---

## ⚕️ Disclaimer
MediScan AI is a research prototype intended for demonstration and hackathon purposes. It is not a substitute for professional medical advice, diagnosis, or treatment.

---
**MediScan AI v1.0** | *Empowering rural healthcare through intelligent multimodal fusion.*
