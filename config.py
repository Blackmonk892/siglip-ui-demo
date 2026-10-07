"""Configuration for the SIGLIP 2 CPU Tester."""

# Using the smallest official SigLIP 2 model for optimal CPU performance
MODEL_ID = "google/siglip2-base-patch16-224"

# Hardcoded to CPU to strictly prevent CUDA/GPU usage
DEVICE = "cpu"

# Default probability threshold for YES/NO classification
DEFAULT_THRESHOLD = 0.5