"""
App.py – convenience launcher at project root.
Run with:  streamlit run App/Main.py
"""
# This file exists so the project root has a recognisable entry point.
# Streamlit should always be pointed at App/Main.py directly.
if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", "App/Main.py"], check=True)
