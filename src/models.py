import os
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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
    _processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
    _model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(device)
    _model.eval()
    return _processor, _model


def encode_text_and_image(
    processor: CLIPProcessor,
    model: CLIPModel,
    text: str,
    image: Image.Image,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
