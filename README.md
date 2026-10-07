# SIGLIP 2 CPU Tester

A minimal, standalone Gradio application for evaluating image-text similarity locally on a CPU using Google's SIGLIP 2 model (`google/siglip2-base-patch16-224`). 

## What This Does
* Encodes an uploaded image into a dense vector.
* Encodes a list of comma-separated text concepts into vectors.
* Calculates independent probabilities (Sigmoid) that the image belongs to each text concept.
* **Does not** draw bounding boxes (this is an image-text matcher, not a YOLO object detector).

## Setup Instructions (Windows PowerShell)

Open PowerShell and run the following commands to isolate and install the project:

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the environment
.venv\Scripts\Activate.ps1

# Note: If Windows blocks activation, run this once as Administrator: 
# Set-ExecutionPolicy Unrestricted -Scope CurrentUser

# 3. Install lightweight CPU dependencies 
pip install -r requirements.txt