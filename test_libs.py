import sys
import time

def test_import(name):
    print(f"Importing {name}...")
    start = time.time()
    try:
        __import__(name)
        print(f"{name} imported in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Failed to import {name}: {e}")

test_import("torch")
test_import("torchvision")
test_import("transformers")
test_import("whisper")
test_import("librosa")
test_import("soundfile")
test_import("numpy")
test_import("PIL")
test_import("streamlit")
test_import("plotly")
test_import("captum")
test_import("cv2")
