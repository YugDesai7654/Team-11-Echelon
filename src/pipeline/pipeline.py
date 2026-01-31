"""
Main Pipeline Orchestrator

Coordinates all pipeline stages for multi-modal misinformation detection.
"""

import time
from typing import Dict, List, Optional
from PIL import Image

from .base import (
    PipelineStage,
    PipelineInput,
    PipelineResult,
    StageResult,
    StageType,
)
from .stages import (
    InputHandlerStage,
    CrossModalDetectorStage,
    ContextDetectorStage,
    SyntheticDetectorStage,
    ExplanationGeneratorStage,
    RobustnessStage,
    EvaluationStage,
)


class MisinformationPipeline:
    """
    Main pipeline for multi-modal misinformation detection.
    
    Orchestrates all 7 deliverables:
    1. Multi-modal input handling
    2. Cross-modal inconsistency detection
    3. Out-of-context media detection
    4. Synthetic media detection
    5. Explanation generation
    6. Robustness checks
    7. Quantitative evaluation
    """
    
    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        """Initialize the pipeline with stages."""
        if stages is None:
            self.stages = self._get_default_stages()
        else:
            self.stages = stages
    
    def _get_default_stages(self) -> List[PipelineStage]:
        """Get default pipeline stages in execution order."""
        return [
            InputHandlerStage(),        # D1: Multi-modal input handling
            SyntheticDetectorStage(),   # D4: Synthetic media detection
            CrossModalDetectorStage(),  # D2: Cross-modal inconsistency
            ContextDetectorStage(),     # D3: Out-of-context detection (SerpApi)
            # RobustnessStage(),          # D6: Robustness checks (PENDING)
            ExplanationGeneratorStage(), # D5: Explanation generation
            # EvaluationStage(),          # D7: Quantitative evaluation (PENDING)
        ]
    
    def run(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        media_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the complete pipeline.
        
        Args:
            text: The claim/caption to analyze
            image: Optional PIL Image object
            media_path: Optional path to image file
            video_path: Optional path to video file
            
        Returns:
            PipelineResult with all detection outputs
        """
        start_time = time.time()
        
        # Create pipeline input
        pipeline_input = PipelineInput(
            text=text,
            image=image,
            media_path=media_path,
            video_path=video_path,
        )
        
        # Load image from path if needed
        if pipeline_input.image is None and media_path:
            try:
                pipeline_input.image = Image.open(media_path).convert("RGB")
            except Exception:
                pass
        
        # Initialize result
        result = PipelineResult()
        stage_results: Dict[StageType, StageResult] = {}
        
        # Execute each stage
        for stage in self.stages:
            print(f"Running stage: {stage.name}...")
            
            # Check if stage should be skipped
            if stage.should_skip(pipeline_input, stage_results):
                print(f"  Skipping {stage.name} (not applicable)")
                continue
            
            # Execute stage
            stage_result = stage.execute(pipeline_input, stage_results)
            stage_results[stage.stage_type] = stage_result
            result.add_stage_result(stage_result)
            
            if stage_result.success:
                print(f"  ✓ {stage.name} completed ({stage_result.execution_time_ms:.0f}ms)")
            else:
                print(f"  ✗ {stage.name} failed: {stage_result.error}")
        
        # Aggregate final results
        self._aggregate_results(result, stage_results)
        
        total_time = (time.time() - start_time) * 1000
        result.raw_data["total_execution_time_ms"] = total_time
        print(f"\nPipeline completed in {total_time:.0f}ms")
        
        return result
    
    def _aggregate_results(self, result: PipelineResult, stage_results: Dict[StageType, StageResult]):
        """Aggregate results from all stages into final output."""
        
        # Get cross-modal score
        cm = stage_results.get(StageType.CROSS_MODAL_DETECTION)
        if cm and cm.success:
            result.cross_modal_score = cm.data.get("similarity", 0.0)
        
        # Get synthetic detection scores
        syn = stage_results.get(StageType.SYNTHETIC_DETECTION)
        if syn and syn.success:
            result.ai_text_probability = syn.data.get("ai_text_probability", 0.0)
            result.ai_image_probability = syn.data.get("ai_image_probability", 0.0)
            result.raw_data["ai_text_result"] = syn.data.get("text_detection", {})
            result.raw_data["ai_image_result"] = syn.data.get("image_detection")
        
        # Get context score
        ctx = stage_results.get(StageType.CONTEXT_DETECTION)
        if ctx and ctx.success:
            result.context_score = ctx.data.get("context_score", 0.0)
        
        # Get robustness score
        rob = stage_results.get(StageType.ROBUSTNESS_CHECK)
        if rob and rob.success:
            result.robustness_score = rob.data.get("robustness_score", 0.0)
        
        # Get final verdict and explanation
        exp = stage_results.get(StageType.EXPLANATION_GENERATION)
        if exp and exp.success:
            result.verdict = exp.data.get("verdict", "Unverified")
            result.truthfulness_score = exp.data.get("truthfulness_score", 0)
            result.explanation = exp.data.get("explanation", "")
            result.evidence = exp.data.get("evidence", [])
        
        # Add CLIP score to raw data for UI
        result.raw_data["clip_score"] = result.cross_modal_score


def detect_misinformation(
    text: str,
    media_path: Optional[str] = None,
    image: Optional[Image.Image] = None
) -> Dict:
    """
    Backward-compatible function that uses the new pipeline.
    
    This maintains the same interface as the original analysis.py function.
    """
    pipeline = MisinformationPipeline()
    result = pipeline.run(text=text, image=image, media_path=media_path)
    
    # Convert to dict format expected by app.py
    output = {
        "verdict": result.verdict,
        "truthfulness_score": result.truthfulness_score,
        "explanation": result.explanation,
        "evidence": result.evidence,
        "clip_score": result.cross_modal_score,
        "ai_text_result": result.raw_data.get("ai_text_result", {}),
        "ai_image_result": result.raw_data.get("ai_image_result"),
    }
    
    return output
