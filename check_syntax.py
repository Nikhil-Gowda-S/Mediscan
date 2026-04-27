"""Syntax-check all project Python files and report any errors."""
import subprocess, sys

files = [
    "App/Models/Image_Encoder.py",
    "App/Models/Fusion_Classifier.py",
    "App/Models/Audio_Encoder.py",
    "App/Components/Explainability.py",
    "App/Components/Image_Uploader.py",
    "App/Components/Audio_Recorder.py",
    "App/Components/Results_Display.py",
    "App/Utils/Model_Loader.py",
    "App/Main.py",
    "App.py",
]

errors = []
for f in files:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", f],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        errors.append(f"FAIL: {f}\n{result.stderr.strip()}")
        print(f"FAIL: {f}")
    else:
        print(f"OK:   {f}")

if errors:
    print("\n=== ERRORS ===")
    for e in errors:
        print(e)
else:
    print("\nAll files compiled successfully!")
