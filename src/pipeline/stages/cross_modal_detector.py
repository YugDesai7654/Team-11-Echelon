"""
Deliverable 2: Detection of cross-modal inconsistencies (caption vs media mismatch)

This stage uses CLIP to detect semantic misalignment between text and images.
"""

import time
from typing import Dict
from dataclasses import dataclass

from ..base import (
    PipelineStage,
    PipelineInput,
    StageResult,
    StageType,
)


# Thresholds for consistency detection
SIMILARITY_CONSISTENT_THRESHOLD = 0.25
SIMILARITY_STRONG_CONSISTENT_THRESHOLD = 0.30


@dataclass
class CrossModalResult:
    """Result of cross-modal consistency detection."""
    verdict: str
    confidence: float
    similarity: float
    explanation: str
    evidence: str


class CrossModalDetectorStage(PipelineStage):
    """
    Stage 2: Cross-Modal Inconsistency Detection
    
    Uses CLIP (Contrastive Language-Image Pre-Training) to detect
    semantic misalignment between text captions and images.
    
    Detects:
    - Caption vs media mismatch
    - Semantic inconsistencies
    - Misleading text-image combinations
    """
    
    def __init__(self):
        self._model = None
        self._processor = None
    
    @property
    def stage_type(self) -> StageType:
        return StageType.CROSS_MODAL_DETECTION
    
    @property
    def name(self) -> str:
        return "Cross-Modal Inconsistency Detector"
    
    @property
    def description(self) -> str:
        return "Detects semantic misalignment between text and images using CLIP"
    
    def should_skip(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> bool:
        """Skip if no image is provided."""
        return not pipeline_input.is_multimodal()
    
    def execute(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> StageResult:
        """
        Detect cross-modal inconsistencies using CLIP.
        """
        start_time = time.time()
        
        try:
            # Import model functions
            from src.models import (
                load_clip_model,
                encode_text_and_image,
                clip_similarity,
            )
            
            # Load CLIP model
            processor, model = load_clip_model()
            
            if processor is None or model is None:
                return StageResult(
                    stage_type=self.stage_type,
                    success=False,
                    error="Failed to load CLIP model",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Encode text and image
            image_embeds, text_embeds = encode_text_and_image(
                processor, model, pipeline_input.text, pipeline_input.image
            )
            
            if image_embeds is None or text_embeds is None:
                return StageResult(
                    stage_type=self.stage_type,
                    success=False,
                    error="Failed to encode inputs",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Calculate similarity
            similarity = clip_similarity(image_embeds, text_embeds)
            
            # Determine verdict and confidence
            result = self._analyze_similarity(similarity)
            
            execution_time = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data={
                    "similarity": round(similarity, 4),
                    "verdict": result.verdict,
                    "confidence": round(result.confidence, 4),
                    "explanation": result.explanation,
                    "evidence": result.evidence,
                    "is_consistent": result.verdict == "Consistent",
                    "is_possible_mismatch": result.verdict != "Consistent",
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return StageResult(
                stage_type=self.stage_type,
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    def _analyze_similarity(self, similarity: float) -> CrossModalResult:
        """Analyze similarity score and generate verdict."""
        if similarity >= SIMILARITY_STRONG_CONSISTENT_THRESHOLD:
            verdict = "Consistent"
            confidence = min(1.0, (similarity - SIMILARITY_CONSISTENT_THRESHOLD) / 
                           (1.0 - SIMILARITY_CONSISTENT_THRESHOLD))
        elif similarity >= SIMILARITY_CONSISTENT_THRESHOLD:
            verdict = "Possible misinformation"
            confidence = 0.5 + 0.5 * (similarity - SIMILARITY_CONSISTENT_THRESHOLD) / (
                SIMILARITY_STRONG_CONSISTENT_THRESHOLD - SIMILARITY_CONSISTENT_THRESHOLD
            )
        else:
            verdict = "Inconsistent"
            confidence = 1.0 - (similarity + 1.0) / (SIMILARITY_CONSISTENT_THRESHOLD + 1.0)
            confidence = max(0.0, min(1.0, confidence))
        
        explanation = self._build_explanation(similarity, verdict)
        evidence = self._build_evidence(similarity, verdict)
        
        return CrossModalResult(
            verdict=verdict,
            confidence=confidence,
            similarity=similarity,
            explanation=explanation,
            evidence=evidence
        )
    
    def _build_explanation(self, similarity: float, verdict: str) -> str:
        """Build human-readable explanation."""
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
    
    def _build_evidence(self, similarity: float, verdict: str) -> str:
        """Build evidence string with thresholds."""
        return (
            f"CLIP text–image similarity: {similarity:.3f} (range -1 to 1). "
            f"Verdict based on threshold: consistent if ≥{SIMILARITY_STRONG_CONSISTENT_THRESHOLD:.2f}, "
            f"possible misinformation if ≥{SIMILARITY_CONSISTENT_THRESHOLD:.2f}, "
            f"inconsistent otherwise."
        )
