from transformers import pipeline
from src.models import get_gemini_response
import json
from PIL import Image

# Initialize the pipeline only once to save resources
try:
    # utilizing a lightweight efficient model for AI text detection
    # roberta-base-openai-detector is a common choice for this
    _text_auth_pipeline = pipeline("text-classification", model="roberta-base-openai-detector")
except Exception as e:
    print(f"Warning: Failed to load AI text detection pipeline: {e}")
    _text_auth_pipeline = None

def detect_ai_text_pipeline(text: str) -> dict:
    """
    Detects if the text is AI-generated using a HuggingFace pipeline.
    Returns:
        dict: {'label': 'Real'|'Fake', 'score': float}
    """
    if not _text_auth_pipeline:
        return {"label": "Unknown", "score": 0.0, "error": "Model not loaded"}
    
    try:
        # Truncate text to 512 tokens roughly (chars/4) to avoid model errors
        truncated_text = text[:2000] 
        result = _text_auth_pipeline(truncated_text)[0]
        
        # logical mapping: label 'Real' usually means Human, 'Fake' means AI
        # specific model mapping needs verification. 
        # For roberta-base-openai-detector: Label_0 is Human, Label_1 is AI (Fake) usually, 
        # or it returns 'Real'/'Fake' labels directly depending on model config.
        # Let's assume standard output first and standardize it.
        
        label = result['label']
        score = result['score']
        
        # Standardize return
        # If model returns LABEL_0 / LABEL_1
        if label == "LABEL_0": # Human
            final_label = "Human" 
            ai_prob = 1 - score
        elif label == "LABEL_1": # AI
            final_label = "AI-Generated"
            ai_prob = score
        elif label.lower() == "real":
            final_label = "Human"
            ai_prob = 1 - score
        elif label.lower() == "fake":
            final_label = "AI-Generated"
            ai_prob = score
        else:
            final_label = label
            ai_prob = score

        return {
            "label": final_label,
            "ai_probability": ai_prob,
            "raw_score": score
        }
    except Exception as e:
        return {"label": "Error", "score": 0.0, "error": str(e)}

def detect_ai_image_gemini(image: Image.Image) -> dict:
    """
    Uses Gemini to analyze an image for synthetic artifacts.
    """
    prompt = (
        "Analyze this image specifically for signs that it is AI-generated/synthetic. "
        "Look for: artificial textures, inconsistent lighting, distorted hands/text/limbs, "
        "hyper-realism typical of Midjourney/DALL-E, or perfect symmetry. "
        "Return a JSON with:"
        "- 'is_ai_generated': boolean,"
        "- 'confidence_score': 0.0 to 1.0 (probability it is AI),"
        "- 'artifacts': list of strings describing specific visual artifacts found (if any)."
    )
    
    generation_config = {"response_mime_type": "application/json"}
    
    try:
        response_text = get_gemini_response(
            prompt, 
            media_content=image, 
            model_name="gemini-flash-lite-latest",
            generation_config=generation_config
        )
        
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(cleaned_text)
        return result_json
        
    except Exception as e:
        return {
            "is_ai_generated": False, 
            "confidence_score": 0.0, 
            "error": str(e)
        }
