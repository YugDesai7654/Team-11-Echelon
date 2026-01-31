"""
Deliverable 1: Multi-modal input handling (text + image and/or video)

This stage handles preprocessing and validation of multi-modal inputs.
"""

import time
from typing import Dict, Optional
from PIL import Image

from ..base import (
    PipelineStage,
    PipelineInput,
    StageResult,
    StageType,
)


class InputHandlerStage(PipelineStage):
    """
    Stage 1: Multi-modal Input Handling
    
    Handles:
    - Text preprocessing and validation
    - Image loading and format conversion
    - Video frame extraction (if applicable)
    - Input validation and normalization
    """
    
    @property
    def stage_type(self) -> StageType:
        return StageType.INPUT_HANDLING
    
    @property
    def name(self) -> str:
        return "Multi-Modal Input Handler"
    
    @property
    def description(self) -> str:
        return "Handles preprocessing and validation of text, image, and video inputs"
    
    def execute(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> StageResult:
        """
        Process and validate multi-modal inputs.
        """
        start_time = time.time()
        
        try:
            processed_data = {}
            
            # Process text input
            text = self._process_text(pipeline_input.text)
            processed_data["text"] = text
            processed_data["text_length"] = len(text)
            processed_data["has_text"] = bool(text.strip())
            
            # Process image input
            if pipeline_input.image is not None:
                image_data = self._process_image(pipeline_input.image)
                processed_data.update(image_data)
                processed_data["has_image"] = True
            elif pipeline_input.media_path:
                image, image_data = self._load_image_from_path(pipeline_input.media_path)
                if image:
                    processed_data.update(image_data)
                    processed_data["loaded_image"] = image
                    processed_data["has_image"] = True
                else:
                    processed_data["has_image"] = False
            else:
                processed_data["has_image"] = False
            
            # Process video input (if applicable)
            if pipeline_input.video_path:
                video_data = self._process_video(pipeline_input.video_path)
                processed_data.update(video_data)
                processed_data["has_video"] = True
            else:
                processed_data["has_video"] = False
            
            # Determine input modality
            processed_data["modality"] = self._determine_modality(processed_data)
            
            execution_time = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data=processed_data,
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
    
    def _process_text(self, text: str) -> str:
        """Clean and normalize text input."""
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _process_image(self, image: Image.Image) -> Dict:
        """Process and validate image input."""
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return {
            "image_width": image.width,
            "image_height": image.height,
            "image_mode": image.mode,
            "image_format": getattr(image, 'format', 'unknown'),
        }
    
    def _load_image_from_path(self, path: str) -> tuple:
        """Load image from file path."""
        try:
            image = Image.open(path).convert("RGB")
            image_data = self._process_image(image)
            return image, image_data
        except Exception as e:
            return None, {"image_load_error": str(e)}
    
    def _process_video(self, video_path: str) -> Dict:
        """Process video input - upload to Gemini and wait for processing."""
        import os
        import google.generativeai as genai
        
        if not os.path.exists(video_path):
             return {
                "video_path": video_path,
                "video_exists": False,
            }

        try:
            print(f"Uploading video {video_path} to Gemini...")
            video_file = genai.upload_file(path=video_path)
            
            while video_file.state.name == "PROCESSING":
                print("Waiting for video processing...")
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                 return {
                    "video_path": video_path,
                    "video_exists": True,
                    "gemini_video_file": None,
                    "video_upload_error": "Gemini video processing failed"
                }

            return {
                "video_path": video_path,
                "video_exists": True,
                "gemini_video_file": video_file
            }
            
        except Exception as e:
            return {
                "video_path": video_path,
                "video_exists": True,
                "gemini_video_file": None,
                "video_upload_error": str(e)
            }
    
    def _determine_modality(self, processed_data: Dict) -> str:
        """Determine the modality of the input."""
        has_text = processed_data.get("has_text", False)
        has_image = processed_data.get("has_image", False)
        has_video = processed_data.get("has_video", False)
        
        if has_text and has_image:
            return "text+image"
        elif has_text and has_video:
            return "text+video"
        elif has_text:
            return "text_only"
        elif has_image:
            return "image_only"
        elif has_video:
            return "video_only"
        else:
            return "none"
