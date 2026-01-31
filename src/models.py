# Models initialization will go here
import google.generativeai as genai
import os

def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("Warning: GOOGLE_API_KEY not found in environment variables.")

def load_clip_model():
    # Placeholder for CLIP model loading
    pass
