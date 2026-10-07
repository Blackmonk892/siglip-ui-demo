"""Model loading and inference logic for SIGLIP 2."""

import torch
from transformers import AutoProcessor, AutoModel
import config

# Global variables to hold the model in memory so it only loads once
_processor = None
_model = None

def load_model():
    """Loads the SIGLIP 2 model and processor once on application startup."""
    global _processor, _model
    
    if _processor is not None and _model is not None:
        return  # Already loaded
        
    print(f"Device: {config.DEVICE}")
    print(f"Loading SIGLIP 2 model ({config.MODEL_ID})...")
    
    # Load processor and model explicitly forcing the CPU device
    _processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    _model = AutoModel.from_pretrained(config.MODEL_ID).to(config.DEVICE)
    _model.eval()
    
    print("Model loaded successfully.")

def analyze_image(image, concepts_text):
    """
    Runs SIGLIP 2 inference on the image and text concepts.
    Returns a list of dictionaries with concept and score.
    """
    load_model()
    
    # 1. Split by comma, trim whitespace, and ignore empty entries
    raw_concepts = [c.strip() for c in concepts_text.split(",")]
    clean_concepts = [c for c in raw_concepts if c]
    
    if not clean_concepts:
        raise ValueError("No valid concepts provided.")
        
    # 2. Format according to SigLIP 2 best practices. 
    # The prompt template "This is a photo of {label}." improves accuracy.
    # We enforce .lower() because SigLIP 2's tokenizer drops case normalization.
    prompts = [f"This is a photo of {c.lower()}." for c in clean_concepts]
    
    print("Running inference on CPU...")
    
    # 3. Prepare inputs natively through the AutoProcessor
    inputs = _processor(
        text=prompts, 
        images=image, 
        padding="max_length", 
        return_tensors="pt"
    ).to(config.DEVICE)
    
    # 4. Run inference efficiently using torch.no_grad()
    with torch.no_grad():
        outputs = _model(**inputs)
        
    # 5. Extract logits and apply sigmoid. 
    # SigLIP uses independent pairwise Sigmoid probabilities, not a global Softmax.
    logits = outputs.logits_per_image
    probs = torch.sigmoid(logits)[0].tolist()
    
    # 6. Pair concepts with scores
    results = []
    for concept, score in zip(clean_concepts, probs):
        results.append({
            "concept": concept,
            "score": score
        })
        
    # Sort descending by score so highest matches appear first
    results.sort(key=lambda x: x["score"], reverse=True)
    return results