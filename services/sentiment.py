import torch
from transformers import pipeline

# Use GPU if available
_device = 0 if torch.cuda.is_available() else -1

_sentiment = pipeline(
    "text-classification", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=_device
)

def analyze_sentiment(text: str):
    """Return sentiment label and score for a given text."""
    if not text or not text.strip():
        return "NEUTRAL", 0.0

    out = _sentiment(text)[0]
    label = out.get("label", "NEUTRAL")
    score = float(out.get("score", 0.0))
    
    # Just return the model labels and score
    return label, round(score, 3)
