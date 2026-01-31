from dataclasses import dataclass
from typing import Optional

from PIL import Image

from src.models import (
    load_clip_model,
    encode_text_and_image,
    clip_similarity,
)


SIMILARITY_CONSISTENT_THRESHOLD = 0.25
SIMILARITY_STRONG_CONSISTENT_THRESHOLD = 0.30


@dataclass
class DetectionResult:
    verdict: str
    confidence: float
    similarity: float
    explanation: str
    evidence: str


def _build_explanation(similarity: float, verdict: str) -> str:
    if verdict == "Consistent":
        return (
            f"The caption and image are semantically aligned (CLIP similarity: {similarity:.3f}). "
            "The text describes content that matches what is shown in the image, with no strong "
            "sign of caption–media mismatch or out-of-context reuse."
        )
    elif verdict == "Possible misinformation":
        return (
            f"The caption and image show weak alignment (CLIP similarity: {similarity:.3f}). "
            "This may indicate caption–media mismatch, out-of-context image reuse, or a caption "
            "that does not accurately describe the image. Further verification is recommended."
        )
    else:
        return (
            f"The caption and image are poorly aligned (CLIP similarity: {similarity:.3f}). "
            "This suggests caption–media mismatch or out-of-context use of the image. "
            "The caption may be misleading or unrelated to the visual content."
        )


def _build_evidence(similarity: float, verdict: str) -> str:
    return (
        f"CLIP text–image similarity: {similarity:.3f} (range -1 to 1). "
        f"Verdict based on threshold: consistent if ≥{SIMILARITY_STRONG_CONSISTENT_THRESHOLD:.2f}, "
        f"possible misinformation if ≥{SIMILARITY_CONSISTENT_THRESHOLD:.2f}, "
        f"inconsistent otherwise."
    )


def detect_cross_modal(
    text: str,
    image: Image.Image,
) -> DetectionResult:
    processor, model = load_clip_model()
    image_embeds, text_embeds = encode_text_and_image(processor, model, text, image)
    similarity = clip_similarity(image_embeds, text_embeds)

    if similarity >= SIMILARITY_STRONG_CONSISTENT_THRESHOLD:
        verdict = "Consistent"
        confidence = min(1.0, (similarity - SIMILARITY_CONSISTENT_THRESHOLD) / (1.0 - SIMILARITY_CONSISTENT_THRESHOLD))
    elif similarity >= SIMILARITY_CONSISTENT_THRESHOLD:
        verdict = "Possible misinformation"
        confidence = 0.5 + 0.5 * (similarity - SIMILARITY_CONSISTENT_THRESHOLD) / (
            SIMILARITY_STRONG_CONSISTENT_THRESHOLD - SIMILARITY_CONSISTENT_THRESHOLD
        )
    else:
        verdict = "Inconsistent"
        confidence = 1.0 - (similarity + 1.0) / (SIMILARITY_CONSISTENT_THRESHOLD + 1.0)
        confidence = max(0.0, min(1.0, confidence))

    explanation = _build_explanation(similarity, verdict)
    evidence = _build_evidence(similarity, verdict)

    return DetectionResult(
        verdict=verdict,
        confidence=round(confidence, 4),
        similarity=round(similarity, 4),
        explanation=explanation,
        evidence=evidence,
    )
