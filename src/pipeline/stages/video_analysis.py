"""
Deliverable: Video Analysis (Holistic)

This stage performs a holistic analysis of video content using Gemini.
It corresponds to the functionality from the `h-video` branch.
"""

import time
import json
from typing import Dict, Any, Optional

import google.generativeai as genai
from ..base import (
    PipelineStage,
    PipelineInput,
    StageResult,
    StageType,
)


class VideoAnalysisStage(PipelineStage):
    """
    Stage: Video Analysis
    
    Performs holistic analysis of video content:
    - Cross-modal inconsistency detection
    - Synthetic media detection
    - Explanation generation
    """
    
    @property
    def stage_type(self) -> StageType:
        return StageType.VIDEO_ANALYSIS
    
    @property
    def name(self) -> str:
        return "Video Analysis (Holistic)"
    
    @property
    def description(self) -> str:
        return "Analyzes video for misinformation using a holistic prompt"
    
    def should_skip(
        self, 
        pipeline_input: PipelineInput, 
        previous_results: Dict[StageType, StageResult]
    ) -> bool:
        """Skip if no video is present."""
        input_result = previous_results.get(StageType.INPUT_HANDLING)
        if not input_result or not input_result.success:
            return True
            
        return not input_result.data.get("has_video", False)

    def execute(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> StageResult:
        """
        Run holistic video analysis.
        """
        start_time = time.time()
        
        try:
            # Get video file from input stage
            input_data = previous_results[StageType.INPUT_HANDLING].data
            gemini_video_file = input_data.get("gemini_video_file")
            
            if not gemini_video_file:
                 return StageResult(
                    stage_type=self.stage_type,
                    success=False,
                    error="No processed video file found",
                    execution_time_ms=0
                )

            # Construct the prompt
            claim_text = pipeline_input.text
            prompt = f"""
            Analyze this video specifically for misinformation regarding the claim: '{claim_text}'
            
            Provide the output in JSON format with the following keys:
            - verdict: (String) "Consistent", "Inconsistent", "Unverified"
            - truthfulness_score: (Integer) 0 to 100
            - explanation: (String) Natural language explanation citing specific visual/audio evidence
            - cross_modal_similarity: (Float) 0.0 to 1.0 (Alignment between video and claim)
            - is_synthetic: (Boolean) True if deepfake or AI-generated
            - confidence: (Float) 0.0 to 1.0
            
            1. Check for cross-modal inconsistencies (Does the video match the text?).
            2. Identify if the media looks AI-generated (Deepfake) or out-of-context.
            3. Provide a natural language explanation.
            """

            # Call Gemini
            model = genai.GenerativeModel('gemini-2.5-flash') # Or appropriate model
            response = model.generate_content([gemini_video_file, prompt], generation_config={"response_mime_type": "application/json"})
            
            # Parse response
            try:
                analysis_result = json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback if not pure JSON (though mime_type should enforce it)
                print(f"Raw response: {response.text}")
                return StageResult(
                     stage_type=self.stage_type,
                     success=False,
                     error="Failed to parse model response",
                     execution_time_ms=(time.time() - start_time) * 1000
                )

            execution_time = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data=analysis_result,
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
