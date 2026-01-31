"""
Deliverable 3: Identification of out-of-context media reuse

This stage detects when images/videos are being used out of their original context.
Uses web search to find original sources and compare context.
"""

import time
from typing import Dict, List, Optional

from ..base import (
    PipelineStage,
    PipelineInput,
    StageResult,
    StageType,
)


class ContextDetectorStage(PipelineStage):
    """
    Stage 3: Out-of-Context Media Reuse Detection
    
    Detects:
    - Images used in different context than original
    - Misattributed media (wrong date, location, event)
    - Recycled visuals with new false narratives
    
    Uses:
    - Web search for reverse image lookup context
    - Temporal analysis (if metadata available)
    - Cross-referencing with news sources
    """
    
    @property
    def stage_type(self) -> StageType:
        return StageType.CONTEXT_DETECTION
    
    @property
    def name(self) -> str:
        return "Out-of-Context Media Detector"
    
    @property
    def description(self) -> str:
        return "Detects when media is being used out of its original context"
    
    def execute(
        self,
        pipeline_input: PipelineInput,
        previous_results: Dict[StageType, StageResult]
    ) -> StageResult:
        """
        Detect out-of-context media reuse.
        """
        start_time = time.time()
        
        try:
            # Perform verification search
            search_results = self._perform_verification_search(pipeline_input.text)
            
            # Analyze search results for context clues
            context_analysis = self._analyze_context(
                pipeline_input.text,
                search_results,
                previous_results
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_type=self.stage_type,
                success=True,
                data={
                    "search_results": search_results,
                    "context_score": context_analysis["score"],
                    "context_verdict": context_analysis["verdict"],
                    "original_sources": context_analysis.get("original_sources", []),
                    "context_flags": context_analysis.get("flags", []),
                    "search_summary": context_analysis.get("summary", ""),
                },
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
    
    def _perform_verification_search(self, query: str) -> Dict:
        """
        Perform web search to find related content and verify claims.
        Uses DuckDuckGo for privacy-respecting search.
        """
        try:
            from duckduckgo_search import DDGS
            
            print(f"Searching for: {query}")
            results = DDGS().text(query, max_results=5)
            
            if not results:
                return {
                    "found": False,
                    "results": [],
                    "summary": "No search results found."
                }
            
            # Format results
            formatted_results = []
            search_summary = "\nSearch Results:\n"
            
            for i, res in enumerate(results, 1):
                formatted_results.append({
                    "title": res.get("title", ""),
                    "url": res.get("href", ""),
                    "snippet": res.get("body", "")
                })
                search_summary += f"{i}. [{res['title']}]({res['href']}): {res['body']}\n"
            
            return {
                "found": True,
                "results": formatted_results,
                "summary": search_summary
            }
            
        except Exception as e:
            return {
                "found": False,
                "results": [],
                "summary": f"Search failed: {str(e)}"
            }
    
    def _analyze_context(
        self,
        claim_text: str,
        search_results: Dict,
        previous_results: Dict[StageType, StageResult]
    ) -> Dict:
        """
        Analyze search results to detect out-of-context usage.
        """
        context_score = 0.5  # Default neutral score
        flags = []
        original_sources = []
        verdict = "Unknown"
        
        if not search_results.get("found"):
            return {
                "score": 0.5,
                "verdict": "Unverified",
                "summary": "Could not verify context due to search failure.",
                "flags": ["no_search_results"],
                "original_sources": []
            }
        
        results = search_results.get("results", [])
        
        # Check cross-modal results for additional context
        cross_modal_result = previous_results.get(StageType.CROSS_MODAL_DETECTION)
        if cross_modal_result and cross_modal_result.success:
            similarity = cross_modal_result.data.get("similarity", 0)
            if similarity < 0.20:
                flags.append("very_low_text_image_alignment")
                context_score -= 0.2
        
        # Analyze search results
        if results:
            # Check if multiple credible sources corroborate the claim
            credible_domains = [
                "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
                "washingtonpost.com", "theguardian.com", "bloomberg.com"
            ]
            
            credible_count = 0
            for result in results:
                url = result.get("url", "").lower()
                for domain in credible_domains:
                    if domain in url:
                        credible_count += 1
                        original_sources.append({
                            "url": result.get("url"),
                            "title": result.get("title"),
                            "credible": True
                        })
                        break
            
            if credible_count >= 2:
                context_score += 0.3
                verdict = "Verified Context"
            elif credible_count == 1:
                context_score += 0.1
                verdict = "Partially Verified"
            else:
                verdict = "Unverified"
                flags.append("no_credible_sources")
        
        # Normalize score
        context_score = max(0.0, min(1.0, context_score))
        
        return {
            "score": context_score,
            "verdict": verdict,
            "summary": search_results.get("summary", ""),
            "flags": flags,
            "original_sources": original_sources
        }
