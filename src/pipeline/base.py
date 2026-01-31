"""
Base classes and data structures for the pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from PIL import Image
from enum import Enum


class StageType(Enum):
    """Enumeration of pipeline stages corresponding to mandatory deliverables."""
    INPUT_HANDLING = "input_handling"
    CROSS_MODAL_DETECTION = "cross_modal_detection"
    CONTEXT_DETECTION = "context_detection"
    SYNTHETIC_DETECTION = "synthetic_detection"
    EXPLANATION_GENERATION = "explanation_generation"
    ROBUSTNESS_CHECK = "robustness_check"
    EVALUATION = "evaluation"


@dataclass
class PipelineInput:
    """
    Unified input structure for the pipeline.
    Deliverable 1: Multi-modal input handling (text + image and/or video)
    """
    text: str
    image: Optional[Image.Image] = None
    video_path: Optional[str] = None
    video_frames: Optional[List[Image.Image]] = None  # Extracted frames from video
    media_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_image(self) -> bool:
        return self.image is not None
    
    def has_video(self) -> bool:
        return self.video_path is not None
    
    def has_video_frames(self) -> bool:
        return self.video_frames is not None and len(self.video_frames) > 0
    
    def is_multimodal(self) -> bool:
        return self.has_image() or self.has_video() or self.has_video_frames()


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_type: StageType
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class PipelineResult:
    """
    Complete result from the pipeline execution.
    Contains results from all stages and final aggregated output.
    """
    # Final outputs
    verdict: str = "Unverified"
    truthfulness_score: int = 0
    explanation: str = ""
    evidence: List[str] = field(default_factory=list)
    
    # Stage-specific results
    stage_results: Dict[StageType, StageResult] = field(default_factory=dict)
    
    # Detailed metrics
    cross_modal_score: float = 0.0
    ai_text_probability: float = 0.0
    ai_image_probability: float = 0.0
    context_score: float = 0.0
    robustness_score: float = 0.0
    
    # Raw data for debugging
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def add_stage_result(self, result: StageResult):
        """Add a stage result to the pipeline result."""
        self.stage_results[result.stage_type] = result
    
    def get_stage_result(self, stage_type: StageType) -> Optional[StageResult]:
        """Get result for a specific stage."""
        return self.stage_results.get(stage_type)
    
    def all_stages_successful(self) -> bool:
        """Check if all executed stages were successful."""
        return all(r.success for r in self.stage_results.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        
        def sanitize_data(data: Any) -> Any:
            """Recursively remove non-serializable objects like PIL Images."""
            if isinstance(data, dict):
                return {k: sanitize_data(v) for k, v in data.items() if k != "video_frames"}
            elif isinstance(data, list):
                if data and isinstance(data[0], Image.Image):
                    return f"<List of {len(data)} Images>"
                return [sanitize_data(i) for i in data]
            elif isinstance(data, Image.Image):
                return f"<Image {data.size} {data.mode}>"
            elif isinstance(data, Enum):
                return data.value
            return data

        results_dict = {}
        for k, v in self.stage_results.items():
            stage_dict = asdict(v)
            stage_dict["stage_type"] = v.stage_type.value # Ensure Enum is serialized
            stage_dict["data"] = sanitize_data(stage_dict["data"])
            results_dict[k.value] = stage_dict

        sanitized_raw = sanitize_data(self.raw_data)

        return {
            "verdict": self.verdict,
            "truthfulness_score": self.truthfulness_score,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "cross_modal_score": self.cross_modal_score,
            "ai_text_probability": self.ai_text_probability,
            "ai_image_probability": self.ai_image_probability,
            "context_score": self.context_score,
            "robustness_score": self.robustness_score,
            "stages_executed": [s.value for s in self.stage_results.keys()],
            "stage_results": results_dict,
            "raw_data": sanitized_raw,
        }


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    Each stage corresponds to a mandatory deliverable.
    """
    
    @property
    @abstractmethod
    def stage_type(self) -> StageType:
        """Return the type of this stage."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the stage."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this stage does."""
        pass
    
    @abstractmethod
    def execute(
        self, 
        pipeline_input: PipelineInput, 
        previous_results: Dict[StageType, StageResult]
    ) -> StageResult:
        """
        Execute the stage.
        
        Args:
            pipeline_input: The input data to process
            previous_results: Results from previously executed stages
            
        Returns:
            StageResult containing the outcome of this stage
        """
        pass
    
    def should_skip(
        self, 
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> bool:
        """
        Determine if this stage should be skipped.
        Override in subclasses for conditional execution.
        """
        return False
