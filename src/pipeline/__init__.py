"""
Multi-Modal Misinformation Detection Pipeline

This package contains modular components for the detection pipeline,
organized by the mandatory deliverables from PS 2.

Modules:
- input_handler: Multi-modal input handling (text + image/video)
- cross_modal_detector: Detection of cross-modal inconsistencies
- context_detector: Identification of out-of-context media reuse
- synthetic_detector: Detection of AI-generated synthetic media
- explanation_generator: Natural language explanation generation
- robustness: Robustness against adversarial perturbations
- evaluation: Quantitative evaluation metrics
- pipeline: Main pipeline orchestrator
"""

from .pipeline import MisinformationPipeline
from .base import PipelineStage, PipelineResult

__all__ = [
    "MisinformationPipeline",
    "PipelineStage",
    "PipelineResult",
]
