"""
Deliverable 6: Robustness against adversarial perturbations
"""

import time
import re
from typing import Dict
from PIL import Image
import numpy as np

from ..base import PipelineStage, PipelineInput, StageResult, StageType


class RobustnessStage(PipelineStage):
    """Stage 6: Robustness Against Adversarial Perturbations"""
    
    @property
    def stage_type(self) -> StageType:
        return StageType.ROBUSTNESS_CHECK
    
    @property
    def name(self) -> str:
        return "Robustness Check"
    
    @property
    def description(self) -> str:
        return "Detects and defends against adversarial perturbations"
    
    def execute(self, pipeline_input: PipelineInput, previous_results: Dict[StageType, StageResult]) -> StageResult:
        start_time = time.time()
        try:
            adversarial_flags = []
            robustness_score = 1.0
            
            text_checks = self._check_text_adversarial(pipeline_input.text)
            adversarial_flags.extend(text_checks["flags"])
            robustness_score *= text_checks["score"]
            
            if pipeline_input.image is not None:
                image_checks = self._check_image_adversarial(pipeline_input.image)
                adversarial_flags.extend(image_checks["flags"])
                robustness_score *= image_checks["score"]
            
            is_adversarial = len(adversarial_flags) > 2 or robustness_score < 0.5
            
            return StageResult(
                stage_type=self.stage_type, success=True,
                data={"robustness_score": round(robustness_score, 4), "adversarial_flags": adversarial_flags, "is_adversarial_likely": is_adversarial},
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return StageResult(stage_type=self.stage_type, success=False, error=str(e), execution_time_ms=(time.time() - start_time) * 1000)
    
    def _check_text_adversarial(self, text: str) -> Dict:
        flags, score = [], 1.0
        if not text:
            return {"flags": [], "score": 1.0}
        
        zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff']
        if any(c in text for c in zero_width):
            flags.append("zero_width_chars"); score *= 0.8
        
        if re.search(r'[\u0400-\u04FF]', text) and re.search(r'[a-zA-Z]', text):
            flags.append("homoglyph_attack"); score *= 0.7
        
        injection = [r'ignore\s+previous', r'you\s+are\s+now', r'disregard\s+all']
        if any(re.search(p, text.lower()) for p in injection):
            flags.append("prompt_injection"); score *= 0.5
        
        return {"flags": flags, "score": max(0.0, min(1.0, score))}
    
    def _check_image_adversarial(self, image: Image.Image) -> Dict:
        flags, score = [], 1.0
        try:
            arr = np.array(image)
            if len(arr.shape) == 3:
                for c in range(arr.shape[2]):
                    if np.std(arr[:,:,c]) < 5:
                        flags.append("uniform_channel"); score *= 0.9; break
        except:
            pass
        return {"flags": flags, "score": max(0.0, min(1.0, score))}
