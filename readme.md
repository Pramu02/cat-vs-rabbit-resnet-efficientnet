# Cats and Rabbit Image Classifier using ResNet18 and EfficientNet-B0

This project requires Python 3.9.

**Setup Instructions:**

1.  **Create a virtual environment (Python 3.9):**
    python3.9 -m venv .venv

2.  **Switch to virtual environment:**
    For Linux/MacOS -> source ./venv/bin/activate
    For Windows -> venv\Scripts\activate.bat

3.  **Install all packages required for the project:**
    pip install -r requirements.txt

Note: script_direct.py uses torchvision’s pretrained ResNet18 and EfficientNet‑B0 models (transfer learning) and script_manual.py contains the manual implementations of ResNet18 and EfficientNet‑B0, trained from scratch.