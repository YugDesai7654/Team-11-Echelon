"""
Deliverable 7: Quantitative evaluation (accuracy, robustness, explanation quality)
"""

import time
from typing import Dict, List
from dataclasses import dataclass, field
from ..base import PipelineStage, PipelineInput, StageResult, StageType


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy_score: float = 0.0
    robustness_score: float = 0.0
    explanation_quality: float = 0.0
    overall_score: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)


class EvaluationStage(PipelineStage):
    """Stage 7: Quantitative Evaluation"""
    
    @property
    def stage_type(self) -> StageType:
        return StageType.EVALUATION
    
    @property
    def name(self) -> str:
        return "Quantitative Evaluation"
    
    @property
    def description(self) -> str:
        return "Computes accuracy, robustness, and explanation quality metrics"
    
    def execute(self, pipeline_input: PipelineInput, previous_results: Dict[StageType, StageResult]) -> StageResult:
        start_time = time.time()
        try:
            metrics = self._compute_metrics(previous_results)
            return StageResult(
                stage_type=self.stage_type, success=True,
                data={"metrics": metrics.__dict__, "overall_score": metrics.overall_score},
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return StageResult(stage_type=self.stage_type, success=False, error=str(e), execution_time_ms=(time.time() - start_time) * 1000)
    
    def _compute_metrics(self, results: Dict[StageType, StageResult]) -> EvaluationMetrics:
        metrics = EvaluationMetrics()
        
        # Robustness score
        rob = results.get(StageType.ROBUSTNESS_CHECK)
        if rob and rob.success:
            metrics.robustness_score = rob.data.get("robustness_score", 0.0)
        
        # Explanation quality
        exp = results.get(StageType.EXPLANATION_GENERATION)
        if exp and exp.success:
            explanation = exp.data.get("explanation", "")
            evidence = exp.data.get("evidence", [])
            metrics.explanation_quality = self._score_explanation(explanation, evidence)
        
        # Cross-modal accuracy proxy
        cm = results.get(StageType.CROSS_MODAL_DETECTION)
        if cm and cm.success:
            metrics.component_scores["cross_modal"] = cm.data.get("confidence", 0.0)
        
        # Synthetic detection confidence
        syn = results.get(StageType.SYNTHETIC_DETECTION)
        if syn and syn.success:
            metrics.component_scores["synthetic"] = 1.0 - syn.data.get("overall_synthetic_score", 0.0)
        
        # Overall score
        scores = [metrics.robustness_score, metrics.explanation_quality] + list(metrics.component_scores.values())
        metrics.overall_score = sum(scores) / max(len(scores), 1)
        
        return metrics
    
    def _score_explanation(self, explanation: str, evidence: List) -> float:
        score = 0.5
        if len(explanation) > 100: score += 0.2
        if len(evidence) >= 2: score += 0.2
        if any(word in explanation.lower() for word in ["because", "evidence", "indicates"]): score += 0.1
        return min(1.0, score)
