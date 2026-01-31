import google.generativeai as genai
import os
from PIL import Image

from tenacity import retry, stop_after_attempt, wait_exponential

def configure_gemini():
    """
    Configures the Gemini API using the API key from environment variables.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment variables.")
        return False
    
    genai.configure(api_key=api_key)
    return True

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def get_gemini_response(prompt_text, media_content=None, model_name="gemini-flash-latest", generation_config=None, tools=None):
    """
    Generates a response from the Gemini model.
    
    Args:
        prompt_text (str): The text prompt for the model.
        media_content (list or object, optional): Image(s) or other media content. 
                                                  Can be a single PIL Image or a list of parts.
        model_name (str): The model version to use. Defaults to "gemini-flash-latest".
        generation_config (dict, optional): Configuration for generation (e.g., json output).
        tools (list, optional): List of tools to use (e.g., Google Search).
        
    Returns:
        str: The generated text response.
    """
    model = genai.GenerativeModel(model_name, tools=tools)
    
    content = [prompt_text]
    if media_content:
        if isinstance(media_content, list):
            content.extend(media_content)
        else:
            content.append(media_content)
    
    response = model.generate_content(content, generation_config=generation_config)
    return response.text

def load_clip_model():
    # Placeholder for CLIP model loading
    pass
