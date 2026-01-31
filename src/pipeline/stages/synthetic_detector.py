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
                # 1. SigLIP: General AI-generated image detection (DALL-E, Midjourney, etc.)
                print("Running AI Image Detection (SigLIP)...")
                ai_image_result = self._detect_ai_image(pipeline_input.image)
                data["ai_image_detection"] = ai_image_result
                data["ai_image_probability"] = ai_image_result.get("confidence_score", 0.0)
                data["is_ai_image"] = ai_image_result.get("is_ai_generated", False)
                
                # 2. ViT Deepfake: Face manipulation detection
                print("Running Deepfake Detection (ViT)...")
                deepfake_result = self._detect_deepfake(pipeline_input.image)
                data["deepfake_detection"] = deepfake_result
                data["deepfake_probability"] = deepfake_result.get("deepfake_probability", 0.0)
                data["is_deepfake"] = deepfake_result.get("is_deepfake", False)
                
                # Combine artifacts from both detectors
                artifacts = []
                if ai_image_result.get("is_ai_generated"):
                    artifacts.append("AI-generated image patterns detected (SigLIP)")
                if deepfake_result.get("is_deepfake"):
                    artifacts.append("Deepfake/face manipulation detected (ViT)")
                data["image_artifacts"] = artifacts
                
                # Combined image detection (take the higher probability)
                data["image_detection"] = {
                    "ai_image": ai_image_result,
                    "deepfake": deepfake_result
                }
            else:
                data["ai_image_detection"] = None
                data["deepfake_detection"] = None
                data["ai_image_probability"] = 0.0
                data["deepfake_probability"] = 0.0
                data["is_ai_image"] = False
                data["is_deepfake"] = False
                data["image_artifacts"] = []
                data["image_detection"] = None
            
            # Detect deepfake in video frames if provided
            if pipeline_input.video_frames and len(pipeline_input.video_frames) > 0:
                print(f"Running Deepfake Detection on {len(pipeline_input.video_frames)} video frames...")
                video_result = self._detect_video_deepfake(pipeline_input.video_frames)
                data["video_detection"] = video_result
                data["deepfake_probability"] = video_result.get("average_ai_probability", 0.0)
                data["is_deepfake"] = video_result.get("is_deepfake", False)
                data["frame_results"] = video_result.get("frame_results", [])
                data["frames_analyzed"] = video_result.get("frames_analyzed", 0)
            else:
                data["video_detection"] = None
                data["deepfake_probability"] = 0.0
                data["is_deepfake"] = False
            
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
        Detect if image is AI-generated using HuggingFace SigLIP-based detector.
        Model: Ateeqq/ai-vs-human-image-detector
        """
        try:
            import torch
            from transformers import AutoImageProcessor, SiglipForImageClassification
            
            MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model and processor
            processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            model = SiglipForImageClassification.from_pretrained(MODEL_ID)
            model.to(device)
            model.eval()
            
            # Preprocess the image
            rgb_image = image.convert("RGB")
            inputs = processor(images=rgb_image, return_tensors="pt").to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Get predictions
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            predicted_prob = probabilities[0, predicted_class_idx].item()
            
            # Get AI probability specifically
            # The model labels are 'ai' and 'hum' (human)
            ai_prob = 0.0
            for i, label in model.config.id2label.items():
                if label.lower() == "ai":
                    ai_prob = probabilities[0, i].item()
                    break
            
            is_ai = predicted_label.lower() == "ai"
            
            return {
                "is_ai_generated": is_ai,
                "confidence_score": ai_prob,
                "predicted_label": predicted_label,
                "predicted_confidence": predicted_prob,
                "artifacts": ["AI-generated patterns detected"] if is_ai else []
            }
            
        except Exception as e:
            print(f"  ⚠️ AI Image Detection Error: {e}")
            return {
                "is_ai_generated": False,
                "confidence_score": 0.0,
                "error": str(e)
            }
    
    def _detect_deepfake(self, image: Image.Image) -> Dict:
        """
        Detect if image contains deepfake/face manipulation using ViT-based detector.
        Model: dima806/deepfake_vs_real_image_detection
        """
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            MODEL_ID = "dima806/deepfake_vs_real_image_detection"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model and processor
            processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
            model.to(device)
            model.eval()
            
            # Preprocess the image
            rgb_image = image.convert("RGB")
            inputs = processor(images=rgb_image, return_tensors="pt").to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Get predictions
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            predicted_prob = probabilities[0, predicted_class_idx].item()
            
            # Get deepfake probability specifically
            # Labels are typically 'Real' and 'Fake'
            deepfake_prob = 0.0
            for i, label in model.config.id2label.items():
                if label.lower() in ["fake", "deepfake"]:
                    deepfake_prob = probabilities[0, i].item()
                    break
            
            is_deepfake = predicted_label.lower() in ["fake", "deepfake"]
            
            return {
                "is_deepfake": is_deepfake,
                "deepfake_probability": deepfake_prob,
                "predicted_label": predicted_label,
                "predicted_confidence": predicted_prob,
            }
            
        except Exception as e:
            print(f"  ⚠️ Deepfake Detection Error: {e}")
            return {
                "is_deepfake": False,
                "deepfake_probability": 0.0,
                "error": str(e)
            }
    
    def _detect_video_deepfake(self, video_frames: list) -> Dict:
        """
        Detect deepfake/AI-generated content in video frames.
        Analyzes each frame and returns averaged results.
        """
        frame_results = []
        total_ai_prob = 0.0
        
        for i, frame in enumerate(video_frames):
            print(f"  Analyzing frame {i+1}/{len(video_frames)}...")
            result = self._detect_ai_image(frame)
            frame_results.append({
                "frame_index": i,
                "ai_probability": result.get("confidence_score", 0.0),
                "is_ai_generated": result.get("is_ai_generated", False),
                "label": result.get("predicted_label", "Unknown")
            })
            total_ai_prob += result.get("confidence_score", 0.0)
        
        # Calculate average probability
        num_frames = len(video_frames)
        avg_ai_prob = total_ai_prob / num_frames if num_frames > 0 else 0.0
        
        # Determine if video is likely a deepfake (threshold: 0.5)
        is_deepfake = avg_ai_prob > 0.5
        
        # Count how many frames were detected as AI-generated
        ai_frame_count = sum(1 for r in frame_results if r["is_ai_generated"])
        
        return {
            "frames_analyzed": num_frames,
            "frame_results": frame_results,
            "average_ai_probability": avg_ai_prob,
            "is_deepfake": is_deepfake,
            "ai_frame_count": ai_frame_count,
            "verdict": "Likely Deepfake" if is_deepfake else "Likely Authentic"
        }
    
    def _compute_overall_score(self, data: Dict) -> float:
        """
        Compute overall synthetic media score.
        Higher score = more likely to be synthetic.
        """
        text_prob = data.get("ai_text_probability", 0.0)
        ai_image_prob = data.get("ai_image_probability", 0.0)
        deepfake_prob = data.get("deepfake_probability", 0.0)
        video_prob = data.get("deepfake_probability", 0.0)  # For video frames
        
        # For images: Take the maximum of AI image and deepfake probabilities
        image_prob = max(ai_image_prob, deepfake_prob)
        
        # Count how many media types we have
        weights = []
        scores = [text_prob]  # Always include text
        weights.append(1.0)
        
        if data.get("image_detection"):
            scores.append(image_prob)
            weights.append(1.0)
        
        if data.get("video_detection"):
            scores.append(video_prob)
            weights.append(1.5)  # Weight video slightly higher
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_sum / total_weight if total_weight > 0 else text_prob

