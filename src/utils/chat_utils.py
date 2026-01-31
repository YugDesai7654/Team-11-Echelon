"""
Chat Utilities for the "Chat with Evidence" feature.

Uses Groq's Llama3-70b model to answer follow-up questions about the
misinformation detection results.
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq


def get_groq_client() -> Groq:
    """
    Initialize and return a Groq client.
    
    Raises:
        ValueError: If GROQ_API_KEY is not set in environment variables.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )
    return Groq(api_key=api_key)


def create_evidence_context(pipeline_result: Dict[str, Any]) -> str:
    """
    Create a structured context string from the pipeline results.
    
    This context is injected into the system prompt so the AI can
    answer questions based on the actual evidence gathered.
    
    Args:
        pipeline_result: The complete pipeline result dictionary.
        
    Returns:
        A formatted string containing all relevant evidence.
    """
    context_parts = []
    
    # Add verdict and overall assessment
    context_parts.append("=== OVERALL VERDICT ===")
    context_parts.append(f"Verdict: {pipeline_result.get('verdict', 'Unknown')}")
    context_parts.append(f"Truthfulness Score: {pipeline_result.get('truthfulness_score', 0)}/100")
    context_parts.append(f"Explanation: {pipeline_result.get('explanation', 'N/A')}")
    
    # Add evidence points
    evidence = pipeline_result.get('evidence', [])
    if evidence:
        context_parts.append("\n=== EVIDENCE POINTS ===")
        for i, e in enumerate(evidence, 1):
            context_parts.append(f"{i}. {e}")
    
    # Add cross-modal analysis
    context_parts.append("\n=== CROSS-MODAL ANALYSIS ===")
    context_parts.append(f"CLIP Similarity Score: {pipeline_result.get('cross_modal_score', 0):.4f}")
    if pipeline_result.get('cross_modal_score', 0) >= 0.30:
        context_parts.append("Assessment: Caption and image are semantically aligned")
    elif pipeline_result.get('cross_modal_score', 0) >= 0.25:
        context_parts.append("Assessment: Weak alignment - possible mismatch detected")
    else:
        context_parts.append("Assessment: Caption does NOT match image content - INCONSISTENT")
    
    # Add synthetic detection results
    context_parts.append("\n=== SYNTHETIC MEDIA ANALYSIS ===")
    context_parts.append(f"AI Text Probability: {pipeline_result.get('ai_text_probability', 0):.1%}")
    context_parts.append(f"AI Image Probability: {pipeline_result.get('ai_image_probability', 0):.1%}")
    
    # Check for deepfake info in raw_data
    raw_data = pipeline_result.get('raw_data', {})
    stage_results = raw_data.get('stage_results', {})
    
    synthetic_data = stage_results.get('synthetic_detection', {})
    if synthetic_data:
        deepfake_prob = synthetic_data.get('deepfake_probability', 0)
        is_deepfake = synthetic_data.get('is_deepfake', False)
        context_parts.append(f"Deepfake Probability: {deepfake_prob:.1%}")
        context_parts.append(f"Deepfake Detected: {'Yes' if is_deepfake else 'No'}")
        
        artifacts = synthetic_data.get('image_artifacts', [])
        if artifacts:
            context_parts.append("Detected Artifacts:")
            for artifact in artifacts[:5]:
                context_parts.append(f"  - {artifact}")
    
    # Add context detection results
    context_data = stage_results.get('context_detection', {})
    if context_data:
        context_parts.append("\n=== OUT-OF-CONTEXT ANALYSIS ===")
        context_parts.append(f"Context Score: {context_data.get('context_score', 0):.2f}")
        context_parts.append(f"Context Verdict: {context_data.get('context_verdict', 'N/A')}")
        
        flags = context_data.get('context_flags', [])
        if flags:
            context_parts.append("Context Flags:")
            for flag in flags:
                context_parts.append(f"  - {flag}")
        
        search_results = context_data.get('search_results', {})
        if search_results.get('found'):
            context_parts.append("Web Search Results Found:")
            for res in search_results.get('results', [])[:3]:
                context_parts.append(f"  - {res.get('title', 'N/A')}: {res.get('url', '#')}")
    
    # Add robustness info
    context_parts.append("\n=== ROBUSTNESS ANALYSIS ===")
    context_parts.append(f"Robustness Score: {pipeline_result.get('robustness_score', 0):.1%}")
    
    return "\n".join(context_parts)


SYSTEM_PROMPT_TEMPLATE = """You are an expert Misinformation Analyst assistant for the Echelon detection system. Your role is to help users understand the analysis results and answer follow-up questions about potential misinformation.

IMPORTANT RULES:
1. You must ONLY answer based on the evidence and analysis provided below.
2. Do not make up information or speculate beyond what the evidence shows.
3. If asked about something not covered in the evidence, clearly state that this information was not part of the analysis.
4. Be helpful, factual, and precise in your responses.
5. When discussing detection results, explain what the scores mean in plain language.
6. If the user asks about why something was flagged, refer to the specific evidence points.

=== ANALYSIS EVIDENCE ===
{evidence_context}

=== END OF EVIDENCE ===

Now, answer the user's questions based solely on this evidence. Be conversational but accurate."""


def get_chat_response(
    messages: List[Dict[str, str]],
    pipeline_result: Dict[str, Any],
    model: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Get a chat response from the Groq API based on the pipeline results.
    
    Args:
        messages: List of chat messages in OpenAI format [{"role": "...", "content": "..."}]
        pipeline_result: The complete pipeline result dictionary for context.
        model: The Groq model to use (default: llama3-70b-8192).
        
    Returns:
        The assistant's response text.
        
    Raises:
        ValueError: If GROQ_API_KEY is not set.
        Exception: If the API call fails.
    """
    client = get_groq_client()
    
    # Create the evidence context from pipeline results
    evidence_context = create_evidence_context(pipeline_result)
    
    # Build the system prompt with injected evidence
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(evidence_context=evidence_context)
    
    # Construct the full messages list with system prompt
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)
    
    # Call the Groq API
    chat_completion = client.chat.completions.create(
        messages=full_messages,
        model=model,
        temperature=0.3,  # Lower temperature for more factual responses
        max_tokens=1024,
    )
    
    return chat_completion.choices[0].message.content


def is_groq_available() -> bool:
    """Check if the Groq API key is configured."""
    return os.getenv("GROQ_API_KEY") is not None
