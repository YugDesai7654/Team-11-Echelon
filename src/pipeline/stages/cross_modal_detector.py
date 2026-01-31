"""
Deliverable 2: Detection of cross-modal inconsistencies (caption vs media mismatch)

This stage uses:
- CLIP for image-text semantic alignment
- Gemini Vision for video context analysis (identifying people, events, locations)
"""

import time
import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

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
    
    For videos, uses Gemini Vision to analyze frame content and
    compare with the claim text for detailed context verification.
    
    Detects:
    - Caption vs media mismatch
    - Semantic inconsistencies
    - Misleading text-image/video combinations
    - Person/event/location mismatches in videos
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
        return "Detects semantic misalignment between text and images/videos using CLIP and Gemini Vision"
    
    def should_skip(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> bool:
        """Skip if no image or video is provided."""
        return not pipeline_input.is_multimodal()
    
    def execute(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> StageResult:
        """
        Detect cross-modal inconsistencies.
        - For images: Use CLIP for semantic alignment
        - For videos: Use Gemini Vision for detailed context analysis
        """
        start_time = time.time()
        
        try:
            # Check if we have video frames to analyze
            video_frames = pipeline_input.video_frames
            if video_frames and len(video_frames) > 0:
                # Use Gemini Vision for video analysis
                return self._analyze_video_with_gemini(
                    pipeline_input.text,
                    video_frames,
                    start_time
                )
            
            # For images, use CLIP
            if pipeline_input.image is not None:
                return self._analyze_image_with_clip(
                    pipeline_input.text,
                    pipeline_input.image,
                    start_time
                )
            
            # No media to analyze
            return StageResult(
                stage_type=self.stage_type,
                success=False,
                error="No image or video frames to analyze",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return StageResult(
                stage_type=self.stage_type,
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    def _analyze_image_with_clip(
        self,
        text: str,
        image: Image.Image,
        start_time: float
    ) -> StageResult:
        """Analyze image-text alignment using CLIP."""
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
            processor, model, text, image
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
                "analysis_type": "image_clip",
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
    
    def _analyze_video_with_gemini(
        self,
        claim_text: str,
        video_frames: List[Image.Image],
        start_time: float
    ) -> StageResult:
        """
        Analyze video frames using Gemini Vision.
        This provides detailed context analysis including:
        - Person identification
        - Event/action description
        - Location identification
        - Comparison with claim text
        """
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return StageResult(
                stage_type=self.stage_type,
                success=False,
                error="GOOGLE_API_KEY not found in environment",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Prepare prompt for video analysis
        analysis_prompt = f"""You are analyzing video frames to verify if they match a given claim/caption.

CLAIM/CAPTION TO VERIFY:
"{claim_text}"

TASK:
Analyze these video frames (taken from start, middle, and end of the video) and determine:

1. **WHO** is visible in the video? Identify any recognizable people (celebrities, athletes, politicians, etc.)
2. **WHAT** is happening in the video? Describe the main action/event.
3. **WHERE** is this taking place? Identify the location if recognizable.
4. **WHEN** - Any temporal indicators (day/night, season, event context)?

5. **VERDICT**: Does the video content MATCH the claim/caption?
   - CONSISTENT: The claim accurately describes the video content
   - INCONSISTENT: The claim does NOT match the video (wrong person, wrong event, wrong location, etc.)
   - UNCERTAIN: Cannot determine with confidence

6. **MISMATCH DETAILS**: If inconsistent, explain EXACTLY what is wrong (e.g., "Video shows Person A but claim says Person B")

7. **CONFIDENCE**: How confident are you in this analysis? (0.0 to 1.0)

Respond in this JSON format:
{{
    "people_identified": ["list of people identified"],
    "action_description": "what is happening",
    "location": "where it is happening",
    "temporal_context": "time-related observations",
    "verdict": "CONSISTENT or INCONSISTENT or UNCERTAIN",
    "mismatch_details": "explanation if inconsistent, or 'None' if consistent",
    "confidence": 0.0 to 1.0,
    "explanation": "detailed natural language explanation for the verdict"
}}"""

        try:
            # Send frames to Gemini for analysis
            response = model.generate_content([analysis_prompt] + video_frames)
            response_text = response.text
            
            # Parse JSON response
            # Clean up response if needed
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            analysis = json.loads(response_text.strip())
            
            # Map verdict to our standard format
            gemini_verdict = analysis.get("verdict", "UNCERTAIN").upper()
            if gemini_verdict == "CONSISTENT":
                verdict = "Consistent"
            elif gemini_verdict == "INCONSISTENT":
                verdict = "Inconsistent"
            else:
                verdict = "Possible misinformation"
            
            confidence = float(analysis.get("confidence", 0.5))
            
            # Build comprehensive explanation
            explanation = analysis.get("explanation", "")
            if analysis.get("mismatch_details") and analysis["mismatch_details"] != "None":
                explanation += f" Mismatch detected: {analysis['mismatch_details']}"
            
            # Build evidence string
            evidence_parts = []
            if analysis.get("people_identified"):
                evidence_parts.append(f"People identified: {', '.join(analysis['people_identified'])}")
            if analysis.get("action_description"):
                evidence_parts.append(f"Action: {analysis['action_description']}")
            if analysis.get("location"):
                evidence_parts.append(f"Location: {analysis['location']}")
            
            evidence = " | ".join(evidence_parts) if evidence_parts else "Video analysis completed"
            
            execution_time = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data={
                    "analysis_type": "video_gemini",
                    "verdict": verdict,
                    "confidence": round(confidence, 4),
                    "explanation": explanation,
                    "evidence": evidence,
                    "is_consistent": verdict == "Consistent",
                    "is_possible_mismatch": verdict != "Consistent",
                    "gemini_analysis": analysis,
                    "frames_analyzed": len(video_frames),
                    "people_identified": analysis.get("people_identified", []),
                    "action_description": analysis.get("action_description", ""),
                    "location": analysis.get("location", ""),
                    "mismatch_details": analysis.get("mismatch_details", ""),
                },
                execution_time_ms=execution_time
            )
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract information from raw response
            execution_time = (time.time() - start_time) * 1000
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data={
                    "analysis_type": "video_gemini",
                    "verdict": "Uncertain",
                    "confidence": 0.5,
                    "explanation": f"Gemini analysis: {response_text[:500]}",
                    "evidence": "Video frames analyzed by Gemini Vision",
                    "is_consistent": False,
                    "is_possible_mismatch": True,
                    "raw_response": response_text,
                    "frames_analyzed": len(video_frames),
                },
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return StageResult(
                stage_type=self.stage_type,
                success=False,
                error=f"Gemini Vision analysis failed: {str(e)}",
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
