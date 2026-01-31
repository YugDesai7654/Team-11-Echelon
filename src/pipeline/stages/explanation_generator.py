"""
Deliverable 5: Natural language explanation generation citing concrete evidence

This stage generates human-understandable explanations for the detection results.
"""

import time
import json
from typing import Dict, List, Optional
from PIL import Image

from ..base import (
    PipelineStage,
    PipelineInput,
    StageResult,
    StageType,
)


class ExplanationGeneratorStage(PipelineStage):
    """
    Stage 5: Natural Language Explanation Generation
    
    Generates:
    - Clear, human-understandable explanations
    - Concrete evidence citations
    - Structured reasoning chains
    - Confidence-qualified statements
    
    Uses Gemini to synthesize signals from all previous stages into
    a coherent, evidence-backed explanation.
    """
    
    @property
    def stage_type(self) -> StageType:
        return StageType.EXPLANATION_GENERATION
    
    @property
    def name(self) -> str:
        return "Explanation Generator"
    
    @property
    def description(self) -> str:
        return "Generates natural language explanations citing concrete evidence"
    
    def execute(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> StageResult:
        """
        Generate natural language explanation using Gemini.
        """
        start_time = time.time()
        
        try:
            from src.models import configure_gemini, get_gemini_response
            
            configure_gemini()
            
            # Build comprehensive prompt with all signals
            prompt = self._build_analysis_prompt(pipeline_input, previous_results)
            
            # Get analysis from Gemini
            generation_config = {"response_mime_type": "application/json"}
            
            response_text = get_gemini_response(
                prompt,
                media_content=pipeline_input.image if pipeline_input.image else None,
                model_name="gemini-flash-lite-latest",
                generation_config=generation_config
            )
            
            # Parse response
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(cleaned_text)
            
            execution_time = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data={
                    "verdict": result_json.get("verdict", "Unverified"),
                    "truthfulness_score": result_json.get("truthfulness_score", 0),
                    "explanation": result_json.get("explanation", ""),
                    "evidence": result_json.get("evidence", []),
                    "raw_response": result_json
                },
                execution_time_ms=execution_time
            )
            
        except json.JSONDecodeError as e:
            execution_time = (time.time() - start_time) * 1000
            return StageResult(
                stage_type=self.stage_type,
                success=False,
                error=f"Failed to parse model response: {str(e)}",
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
    
    def _build_analysis_prompt(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> str:
        """Build comprehensive prompt incorporating all detection signals."""
        
        is_multimodal = pipeline_input.is_multimodal()
        
        # Start with role and task
        if is_multimodal:
            prompt = (
                "You are an expert Multi-Modal Misinformation & AI Detection Assistant. "
                "Your task is to analyze the provided claim and media to determine its truthfulness and origin.\n"
            )
        else:
            prompt = (
                "You are an expert Fact-Checker and AI Text Detection Assistant. "
                "Your task is to analyze the provided text to determine its truthfulness and origin.\n"
            )
        
        # Add system detection signals
        prompt += "\n--- SYSTEM DETECTION SIGNALS ---\n"
        
        # Synthetic detection signals
        synthetic_result = previous_results.get(StageType.SYNTHETIC_DETECTION)
        if synthetic_result and synthetic_result.success:
            text_prob = synthetic_result.data.get("ai_text_probability", 0)
            text_label = synthetic_result.data.get("text_label", "Unknown")
            prompt += f"AI Text Probability: {text_prob:.2%} (Label: {text_label})\n"
            
            if is_multimodal:
                image_prob = synthetic_result.data.get("ai_image_probability", 0)
                is_ai_image = synthetic_result.data.get("is_ai_image", False)
                artifacts = synthetic_result.data.get("image_artifacts", [])
                
                prompt += f"AI Image Probability: {image_prob:.2%} (Is AI: {is_ai_image})\n"
                if artifacts:
                    prompt += f"Detected Visual Artifacts: {', '.join(artifacts)}\n"
        
        # Cross-modal signals (if multimodal)
        if is_multimodal:
            cross_modal_result = previous_results.get(StageType.CROSS_MODAL_DETECTION)
            if cross_modal_result and cross_modal_result.success:
                similarity = cross_modal_result.data.get("similarity", 0)
                cm_verdict = cross_modal_result.data.get("verdict", "Unknown")
                cm_explanation = cross_modal_result.data.get("explanation", "")
                
                prompt += (
                    f"\nContext from CLIP Analysis:\n"
                    f"- Semantic Similarity: {similarity:.2f}\n"
                    f"- Cross-Modal Verdict: {cm_verdict}\n"
                    f"- Automatic Explanation: {cm_explanation}\n"
                )
                
                if similarity < 0.20:
                    prompt += " [(IMPORTANT) The CLIP score is VERY LOW. The image likely has nothing to do with the text.]\n"
        
        # Context/search signals
        context_result = previous_results.get(StageType.CONTEXT_DETECTION)
        if context_result and context_result.success:
            search_summary = context_result.data.get("search_summary", "")
            if search_summary:
                prompt += f"\n\nContext from Web Search (Verification Data):\n{search_summary}\n"
        
        # Add claim
        prompt += f"\n\nClaim to Verify: {pipeline_input.text}\n"
        
        # Task instructions
        if is_multimodal:
            prompt += (
                "Task Instructions:\n"
                "1. **Origin Analysis**: Based on the System Detection Signals and your own analysis, explicitly state if the text or image is AI-generated. NOTE: A claim can be AI-generated but still Factually TRUE.\n"
                "2. **Identity & Context Verification**: Use the Web Search Results to verify the identities, events, and facts. Check for hallucinations common in AI text.\n"
                "3. **Cross-Modal Consistency**: Does the image support the claim? If it's an AI image, is it being passed off as real footage?\n"
                "4. **Truthfulness Assessment**: Final verdict on the TRUTH of the claim. \n"
                "   - If Text is AI-generated + Factually Correct -> Verdict: 'Real' (with note 'AI-Drafted').\n"
                "   - If Image is AI-generated + Passed as Real -> Verdict: 'Fake' or 'Misleading'.\n"
                "5. **Explanation**: Clear reasoning.\n"
            )
        else:
            prompt += (
                "Task Instructions:\n"
                "1. **Origin Analysis**: Based on the AI Text Probability and wording, determine if the text is likely AI-generated. NOTE: A claim can be AI-generated but still Factually TRUE.\n"
                "2. **Fact Verification**: Use the Web Search Results to verify the identities, events, and facts. Check for hallucinations common in AI text.\n"
                "3. **Truthfulness Assessment**: Final verdict on the TRUTH of the claim. \n"
                "   - If Text is AI-generated + Factually Correct -> Verdict: 'Real' (with note 'AI-Drafted').\n"
                "   - If Text is Factually FALSE -> Verdict: 'Fake' or 'Misleading'.\n"
                "4. **Explanation**: Clear reasoning focusing on factual accuracy.\n"
            )
        
        # Output format
        prompt += (
            "\nOutput Format: Return the result as a raw JSON object (no markdown formatting) with the following keys:\n"
            "- 'verdict': One of ['Real', 'Fake', 'Misleading', 'Out-of-Context', 'Unverified', 'AI-Generated (True)', 'AI-Generated (Fake)']\n"
            "- 'truthfulness_score': An integer from 0 to 100 (100 = Definitive Truth, 0 = Definitive Lie/Fake)\n"
            "- 'explanation': A natural language paragraph explaining the reasoning.\n"
            "- 'evidence': A list of bullet points citing specific inconsistencies or facts found via search.\n"
        )
        
        if is_multimodal:
            prompt += "\nMedia is attached below.\n"
        
        return prompt
