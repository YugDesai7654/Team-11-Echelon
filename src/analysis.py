"""
Main Analysis Pipeline - Simplified

Uses CLIP for cross-modal consistency detection only.
"""

from typing import Any, Dict, Optional

from PIL import Image

from src.clip_detector import detect_cross_modal


def detect_misinformation(
    text: str,
    media_path: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """
    Detect misinformation using CLIP cross-modal analysis.
    
    Args:
        text: Caption or claim text
        media_path: Path to image file (optional if image provided)
        image: PIL Image object (optional if media_path provided)
    
    Returns:
        Dict with verdict, confidence, similarity, and explanation
    """
    # Validate input
    if image is None and media_path is None:
        return {
            "verdict": "Error",
            "confidence": 0.0,
            "explanation": "Provide either an image file path or a PIL Image.",
            "similarity": None,
        }
    
    # Load image if path provided
    if image is None:
        try:
            image = Image.open(media_path).convert("RGB")
        except Exception as e:
            return {
                "verdict": "Error",
                "confidence": 0.0,
                "explanation": f"Could not load image: {e}",
                "similarity": None,
            }
    
    # Run CLIP cross-modal detection
    result = detect_cross_modal(text, image)
    
    return {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "similarity": result.similarity,
        "explanation": result.explanation,
        "evidence": result.evidence,
    }
