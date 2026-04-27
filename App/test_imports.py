import sys
import os
from pathlib import Path

# Simulate streamlit path behavior
sys.path.append(str(Path(__file__).parent))

print("Testing imports...")
try:
    print("Importing ModelLoader...")
    from Utils.Model_Loader import ModelLoader
    print("ModelLoader imported")
    print("Importing ImageUploader...")
    from Components.Image_Uploader import ImageUploader
    print("ImageUploader imported")
    print("Importing AudioRecorder...")
    from Components.Audio_Recorder import AudioRecorder
    print("AudioRecorder imported")
    print("Importing ResultDisplay...")
    from Components.Results_Display import ResultDisplay
    print("ResultDisplay imported")
    print("Importing Explainability...")
    from Components.Explainability import GradcamOverlay
    print("Explainability imported")
    print("Importing ImageEncoder...")
    from Models.Image_Encoder import ImageEncoder
    print("ImageEncoder imported")
    print("All imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
