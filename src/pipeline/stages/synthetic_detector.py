"""
Deliverable 4: Detection of AI-generated synthetic media (text, images, deepfakes)

This stage detects artificially generated content across multiple modalities.
"""

import time
import json
from typing import Dict, Optional
from PIL import Image

from ..base import (
    PipelineStage,
    PipelineInput,
    StageResult,
    StageType,
)


# Global pipeline instance for AI text detection
_text_auth_pipeline = None


def _load_text_detector():
    """Lazy load the text detection pipeline."""
    global _text_auth_pipeline
    if _text_auth_pipeline is None:
        try:
            from transformers import pipeline
            _text_auth_pipeline = pipeline(
                "text-classification",
                model="roberta-base-openai-detector"
            )
        except Exception as e:
            print(f"Warning: Failed to load AI text detection pipeline: {e}")
    return _text_auth_pipeline


class SyntheticDetectorStage(PipelineStage):
    """
    Stage 4: AI-Generated Synthetic Media Detection
    
    Detects:
    - AI-generated text (GPT, Claude, etc.)
    - AI-generated images (DALL-E, Midjourney, Stable Diffusion)
    - Deepfake videos (if video input provided)
    
    Uses:
    - RoBERTa-based OpenAI detector for text
    - Gemini vision for image artifact detection
    - Pattern analysis for synthetic media fingerprints
    """
    
    @property
    def stage_type(self) -> StageType:
        return StageType.SYNTHETIC_DETECTION
    
    @property
    def name(self) -> str:
        return "Synthetic Media Detector"
    
    @property
    def description(self) -> str:
        return "Detects AI-generated text, images, and deepfakes"
    
    def execute(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> StageResult:
        """
        Detect AI-generated synthetic media.
        """
        start_time = time.time()
        
        try:
            data = {}
            
            # Always detect AI text
            print("Running AI Text Detection...")
            text_result = self._detect_ai_text(pipeline_input.text)
            data["text_detection"] = text_result
            data["ai_text_probability"] = text_result.get("ai_probability", 0.0)
            data["text_label"] = text_result.get("label", "Unknown")
            
            # Detect AI image if image is provided
            if pipeline_input.image is not None:
                print("Running AI Image Detection (Gemini)...")
                image_result = self._detect_ai_image(pipeline_input.image)
                data["image_detection"] = image_result
                data["ai_image_probability"] = image_result.get("confidence_score", 0.0)
                data["is_ai_image"] = image_result.get("is_ai_generated", False)
                data["image_artifacts"] = image_result.get("artifacts", [])
            else:
                data["image_detection"] = None
                data["ai_image_probability"] = 0.0
                data["is_ai_image"] = False
            
            # Compute overall synthetic score
            data["overall_synthetic_score"] = self._compute_overall_score(data)
            
            execution_time = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data=data,
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
    
    def _detect_ai_text(self, text: str) -> Dict:
        """
        Detect if text is AI-generated using RoBERTa-based detector.
        """
        pipeline = _load_text_detector()
        
        if not pipeline:
            return {
                "label": "Unknown",
                "ai_probability": 0.0,
                "error": "Model not loaded"
            }
        
        try:
            # Truncate text to avoid model errors (roughly 512 tokens)
            truncated_text = text[:2000]
            result = pipeline(truncated_text)[0]
            
            label = result['label']
            score = result['score']
            
            # Standardize label mapping
            if label == "LABEL_0":  # Human
                final_label = "Human"
                ai_prob = 1 - score
            elif label == "LABEL_1":  # AI
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
            return {
                "label": "Error",
                "ai_probability": 0.0,
                "error": str(e)
            }
    
    def _detect_ai_image(self, image: Image.Image) -> Dict:
        """
        Detect if image is AI-generated using Gemini vision analysis.
        """
        try:
            from src.models import get_gemini_response, configure_gemini
            
            configure_gemini()
            
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
            
            response_text = get_gemini_response(
                prompt,
                media_content=image,
                model_name="gemini-flash-lite-latest",
                generation_config=generation_config
            )
            
            # Clean response
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(cleaned_text)
            
            return result_json
            
        except Exception as e:
            return {
                "is_ai_generated": False,
                "confidence_score": 0.0,
                "error": str(e)
            }
    
    def _compute_overall_score(self, data: Dict) -> float:
        """
        Compute overall synthetic media score.
        Higher score = more likely to be synthetic.
        """
        text_prob = data.get("ai_text_probability", 0.0)
        image_prob = data.get("ai_image_probability", 0.0)
        
        # Weight text and image equally if both present
        if data.get("image_detection"):
            return (text_prob + image_prob) / 2
        else:
            return text_prob
