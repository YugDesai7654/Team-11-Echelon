"""
Deliverable 3: Identification of out-of-context media reuse

This stage detects when images/videos are being used out of their original context.
Uses SerpApi's Google Reverse Image Search + Gemini AI to compare contexts.
"""

import os
import time
import tempfile
import requests
import base64
from datetime import datetime
from typing import Dict, List, Optional
from dateutil import parser as date_parser
from PIL import Image
import google.generativeai as genai

from ..base import (
    PipelineStage,
    PipelineInput,
    StageResult,
    StageType,
)


class ContextDetectorStage(PipelineStage):
    """
    Stage 3: Out-of-Context Media Reuse Detection
    
    Uses SerpApi for Reverse Image Search and Gemini AI for Semantic Comparison.
    """
    
    def __init__(self, serpapi_key: Optional[str] = None):
        self.api_key = serpapi_key or os.getenv("SERPAPI_KEY")
        
    @property
    def stage_type(self) -> StageType:
        return StageType.CONTEXT_DETECTION
    
    @property
    def name(self) -> str:
        return "Out-of-Context Media Detector"
    
    @property
    def description(self) -> str:
        return "Detects out-of-context reuse by comparing claim with visual search results using AI"
    
    def should_skip(self, pipeline_input: PipelineInput, previous_results: Dict[StageType, StageResult]) -> bool:
        if not self.api_key:
            print("  ⚠️ SERPAPI_KEY not set - skipping context detection")
            return True
        if not pipeline_input.has_image() and not pipeline_input.media_path:
            return True
        return False
    
    def execute(self, pipeline_input: PipelineInput, previous_results: Dict[StageType, StageResult]) -> StageResult:
        start_time = time.time()
        try:
            image = pipeline_input.image
            if image is None and pipeline_input.media_path:
                image = Image.open(pipeline_input.media_path).convert("RGB")
            
            if image is None:
                return StageResult(self.stage_type, False, error="No image available")
            
            # 1. Get Image URL
            image_url = self._upload_image_temp(image)
            if not image_url:
                return StageResult(self.stage_type, False, error="Upload failed")
            
            # 2. Reverse Search
            search_results = self._reverse_image_search(image_url)
            
            # 3. AI Verification
            analysis = self._ai_analyze_context(search_results, pipeline_input.text)
            
            execution_time = (time.time() - start_time) * 1000
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data=analysis,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            return StageResult(self.stage_type, False, error=str(e))

    def _upload_image_temp(self, image: Image.Image) -> Optional[str]:
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image.save(temp_file.name, "JPEG", quality=85)
        temp_file.close()
        try:
            with open(temp_file.name, "rb") as f:
                response = requests.post(
                    "https://freeimage.host/api/1/upload",
                    data={"key": "6d207e02198a847aa98d0a2a901485a5"},
                    files={"source": f}
                )
            if response.status_code == 200:
                return response.json().get("image", {}).get("url")
            return None
        finally:
            os.unlink(temp_file.name)

    def _reverse_image_search(self, image_url: str) -> Dict:
        from serpapi import GoogleSearch
        import json
        
        params = {"engine": "google_lens", "url": image_url, "api_key": self.api_key}
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Console log the SerpApi response for debugging
        print("\n" + "="*60)
        print("🔍 SERPAPI REVERSE IMAGE SEARCH RESULTS")
        print("="*60)
        
        # Log visual matches
        visual_matches = results.get("visual_matches", [])
        print(f"\n📸 Found {len(visual_matches)} Visual Matches:")
        for i, match in enumerate(visual_matches[:5], 1):
            print(f"  {i}. {match.get('title', 'No title')}")
            print(f"     Source: {match.get('source', 'Unknown')}")
            print(f"     Link: {match.get('link', 'N/A')}")
            if match.get('date'):
                print(f"     Date: {match.get('date')}")
            print()
        
        # Log knowledge graph if present
        knowledge_graph = results.get("knowledge_graph", [])
        if knowledge_graph:
            print(f"\n📚 Knowledge Graph:")
            print(f"   {json.dumps(knowledge_graph[:2], indent=2)}")
        
        # Log any text detected
        text_results = results.get("text_results", [])
        if text_results:
            print(f"\n📝 Text Detected in Image:")
            for text in text_results[:3]:
                print(f"   - {text.get('text', 'N/A')}")
        
        print("="*60 + "\n")
        
        return results

    def _ai_analyze_context(self, search_results: Dict, claim: str) -> Dict:
        """Uses Gemini to compare the user's claim with search results."""
        matches = search_results.get("visual_matches", [])
        if not matches:
            return {
                "context_score": 0.6,
                "is_out_of_context": False,
                "verdict": "No Prior Usage Found",
                "summary": "This image seems original or very new."
            }

        # Format match data for Gemini
        search_data = ""
        for m in matches[:8]:
            search_data += f"- Title: {m.get('title')}\n  Source: {m.get('source')}\n  Date: {m.get('date', 'Unknown')}\n\n"

        prompt = f"""
        You are a Fact-Checking AI. I will give you a USER CLAIM and actual SEARCH RESULTS for an image.
        Detect if the image is being used OUT OF CONTEXT.
        
        USER CLAIM: "{claim}"
        
        SEARCH RESULTS FOR THIS IMAGE:
        {search_data}
        
        ANALYSIS RULES:
        1. If user says "Breaking news today" but search shows image is from years ago, flag as OUT OF CONTEXT.
        2. If user mentions a location (e.g. Mumbai) but search says it's from another place (e.g. Bangladesh), flag as OUT OF CONTEXT.
        3. If search results consistently show a different event than the claim, flag as OUT OF CONTEXT.
        
        RESPONSE FORMAT:
        Verdict: [OUT OF CONTEXT / VERIFIED / SUSPICIOUS]
        Score: [0-100, where 0 is fake context and 100 is perfectly verified]
        Reason: [One sentence explaining why]
        """

        try:
            from src.models import configure_gemini, get_gemini_response
            if not configure_gemini():
                return {
                    "context_score": 0.5, 
                    "context_verdict": "AI Configuration Failed", 
                    "search_summary": "Gemini key missing.",
                    "context_flags": ["api_config_failed"]
                }

            text = get_gemini_response(prompt, model_name="gemini-flash-lite-latest")
            
            # Simple parsing of AI response
            is_ooc = "OUT OF CONTEXT" in text.upper()
            is_suspicious = "SUSPICIOUS" in text.upper()
            
            if is_ooc:
                verdict = "⚠️ OUT OF CONTEXT"
            elif is_suspicious:
                verdict = "⚠️ SUSPICIOUS CONTEXT"
            else:
                verdict = "✅ Verified Context"
            
            # Improved flag detection from AI response
            import re
            score_match = re.search(r'Score:\s*(\d+)', text)
            score = int(score_match.group(1)) / 100.0 if score_match else 0.5
            
            flags = []
            
            # Use specific human-friendly names that the UI likes
            if is_ooc:
                flags.append("Out-of-Context Media")
            
            text_lower = text.lower()
            if "date" in text_lower or "year" in text_lower or "old" in text_lower or "temporal" in text_lower:
                flags.append("Date Mismatch")
            if "location" in text_lower or "place" in text_lower or "mumbai" in text_lower or "surat" in text_lower:
                flags.append("Location Mismatch")
            if "event" in text_lower or "different context" in text_lower:
                flags.append("Event Mismatch")
            if "recycled" in text_lower or "reused" in text_lower:
                flags.append("Recycled Content")

            # Fallback if AI says it's OOC but no specific flag was caught
            if is_ooc and not flags:
                flags.append("Context Mismatch")

            return {
                "context_score": score,
                "is_out_of_context": is_ooc,
                "context_verdict": verdict,
                "search_summary": text.split("Reason:")[-1].strip(),
                "full_ai_analysis": text,
                "context_flags": flags,
                "visual_matches": matches[:3]
            }
        except Exception as e:
            return {
                "context_score": 0.5,
                "is_out_of_context": False,
                "context_verdict": "AI Analysis Failed",
                "search_summary": f"Error: {str(e)}",
                "context_flags": ["Analysis Error"],
                "visual_matches": matches[:3]
            }
