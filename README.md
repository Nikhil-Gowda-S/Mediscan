# MediScan AI - The 60-Second Diagnosis Layer

## Overview
MediScan AI is a multimodal AI application designed for rural healthcare, providing rapid diagnostic support using chest X-rays, recorded patient symptoms, and vital signs.

## Features
- **X-Ray Analysis**: DenseNet-121 backbone trained on medical imagery.
- **Symptom Transcription**: OpenAI Whisper for speech-to-text.
- **Clinical NLP**: BioBERT for processing medical symptoms.
- **Multimodal Fusion**: Cross-modal attention mechanism to fuse image, audio, and vital features.
- **Explainability**: Grad-CAM visualization to highlight diagnostic regions in X-rays.
- **Uncertainty Quantification**: Monte Carlo Dropout for gauging diagnostic confidence.

## Project Structure
```
Mediscan-Ai/
├── App/
│   ├── Main.py
│   ├── Components/
│   │   ├── Image_Uploader.py
│   │   ├── Audio_Recorder.py
│   │   ├── Results_Display.py
│   │   └── Explainability.py
│   ├── Models/
│   │   ├── Image_Encoder.py
│   │   ├── Audio_Encoder.py
│   │   ├── Fusion_Classifier.py
│   └── Utils/
│       └── Model_Loader.py
├── Data/
│   └── Samples/
├── Requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Getting Started

### 📦 Input Requirements
To test the project's multimodal capabilities, prepare the following inputs:
- **Chest X-Ray**: A grayscale image (JPG/PNG). Recommended size `224x224`. 
  - *Demo sample*: Search for "Chest X-Ray Normal" or "Pneumonia X-Ray" on Google Images.
- **Microphone Input**: Use the "Record Symptoms" button to speak symptoms like *"I have a sharp pain in my chest and a dry cough"*.
- **Patient Vitals**: You will need to input numeric values for Temperature, Blood Pressure, and SpO2.

### 🛠️ Local Setup
1. **Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run App/Main.py
   ```

### 🐳 Docker (Recommended for Consistency)
```bash
docker-compose up --build
```

### ⚙️ Troubleshooting
- **First Run**: The app will download model weights (~500MB+). Ensure a stable internet connection.
- **GPU**: If a CUDA-compatible GPU is available, the app will automatically use it for 10x faster inference.
- **Mock Mode**: If a model fails to load, the app switches to **Demo Mode** with mock encoders to ensure the UI remains interactive.

## ✨ Advanced Features Added
- **Patient Portal**: Simulated patient records and vital history tracking.
- **High-Precision Fusion**: Cross-attention mechanism between clinical text and visual features.
- **Uncertainty Quantification**: Monte Carlo Dropout for gauging diagnostic risk.
- **Explainable AI (XAI)**: Grad-CAM heatmap visualization.
