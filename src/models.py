# Models initialization will go here
from google import genai
import os

def configure_gemini():
    """
    Returns a configured Gemini client using the new google-genai SDK.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    else:
        print("Warning: GOOGLE_API_KEY not found in environment variables.")
        return None

def load_clip_model():
    # Placeholder for CLIP model loading
    pass
