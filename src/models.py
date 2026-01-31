import os
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional, Tuple
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load environment variables
load_dotenv()

# --- Gemini Configuration ---
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
def get_gemini_response(prompt_text, media_content=None, model_name="gemini-flash-lite-latest", generation_config=None, tools=None):
    """
    Generates a response from the Gemini model.
    Retries on transient errors (like 429).
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

# --- CLIP Configuration ---

_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_processor: Optional[CLIPProcessor] = None
_model: Optional[CLIPModel] = None
_device: Optional[str] = None


def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load_clip_model() -> Tuple[CLIPProcessor, CLIPModel]:
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model
    device = get_device()
    try:
        _processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        _model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(device)
        _model.eval()
    except Exception as e:
        print(f"Warning: Failed to load CLIP model. {e}")
        return None, None
        
    return _processor, _model


def encode_text_and_image(
    processor: CLIPProcessor,
    model: CLIPModel,
    text: str,
    image: Image.Image,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if processor is None or model is None:
        return None, None
        
    device = next(model.parameters()).device
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    return image_embeds, text_embeds


def clip_similarity(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> float:
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    sim = (image_embeds @ text_embeds.T).item()
    return float(sim)
