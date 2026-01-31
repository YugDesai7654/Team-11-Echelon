from src.models import configure_gemini, get_gemini_response
import json
from duckduckgo_search import DDGS
from src.clip_detector import detect_cross_modal
from typing import Optional, Dict, Any
from PIL import Image

def perform_verification_search(query):
    """
    Performs a DuckDuckGo search to verify the claim.
    Returns a summary of the top results.
    """
    try:
        print(f"Searching for: {query}")
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No search results found."
        
        search_summary = "\nSearch Results:\n"
        for i, res in enumerate(results, 1):
            search_summary += f"{i}. [{res['title']}]({res['href']}): {res['body']}\n"
        
        return search_summary
    except Exception as e:
        return f"Search failed: {str(e)}"

def detect_misinformation(
    text: str, 
    media_path: Optional[str] = None, 
    image: Optional[Image.Image] = None, 
    clip_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyzes the text and media to detect potential misinformation, using Gemini 
    and incorporating CLIP similarity context.
    
    Args:
        text (str): The claim or caption to analyze.
        media_path (str, optional): Path to image file.
        image (PIL.Image, optional): The image object.
        clip_score (float, optional): Legacy argument, calculated automatically if not provided.
        
    Returns:
        dict: A dictionary containing the structured analysis results.
    """
    # 0. Load Image (Unified logic)
    if image is None and media_path:
        try:
            image = Image.open(media_path).convert("RGB")
        except Exception:
            pass # Handle gracefully later
            
    # 1. Run CLIP Cross-Modal Analysis (Teammate's Logic)
    clip_context = ""
    real_clip_score = 0.0
    
    if image:
        try:
            print("Running CLIP analysis...")
            clip_result = detect_cross_modal(text, image)
            real_clip_score = clip_result.similarity
            clip_context = (
                f"\nContext from CLIP Analysis:\n"
                f"- Semantic Similarity: {clip_result.similarity:.2f}\n"
                f"- Cross-Modal Verdict: {clip_result.verdict}\n"
                f"- Automatic Explanation: {clip_result.explanation}\n"
            )
        except Exception as e:
            print(f"CLIP analysis failed: {e}")
            clip_context = "\nCLIP analysis could not be performed.\n"

    # Use the passed clip_score if CLIP failed or wasn't run, otherwise use real one
    final_clip_score = real_clip_score if real_clip_score > 0 else (clip_score or 0.0)

    # 2. Ensure Gemini is configured
    configure_gemini()
    
    # 3. Perform Search Grounding (My Logic)
    search_context = perform_verification_search(text)
    
    # 4. Construct the prompt with CLIP context and specific analysis instructions
    prompt = (
        f"You are an expert Multi-Modal Misinformation Detection AI. "
        f"Your task is to analyze the provided claim and media to determine its truthfulness. "
    )
    
    prompt += clip_context
    
    if final_clip_score < 0.20:
         prompt += " [(IMPORTANT) The CLIP score is VERY LOW. The image likely has nothing to do with the text.]"
            
    # Inject search results into prompt
    prompt += f"\n\nContext from Web Search (Verification Data):\n{search_context}\n"
    
    prompt += (
        f"\n\nClaim to Verify: {text}\n"
        f"Task Instructions:\n"
        f"1. **Visual Analysis**: detailedly describe what is happening in the image/video.\n"
        f"2. **Identity & Context Verification (CRITICAL)**: Use the provided Web Search Results to verify the identities of any people in the image/claim. Check for public profiles (LinkedIn, Instagram, etc.) to confirm if the person is a local resident or public figure as claimed.\n"
        f"3. **Cross-Modal Consistency**: Does the image actually support the specific details in the claim? (e.g., location, time, people, events)? Use the CLIP context as a hint.\n"
        f"4. **Truthfulness Assessment**: Based on your knowledge and the provided Search Results, is this claim true? If it refers to a real event, is the image from that event or is it out-of-context (reused)?\n"
        f"5. **Explanation**: Provide a clear, step-by-step reasoning for your verdict, citing the search results where applicable.\n"
        f"\nOutput Format: Return the result as a raw JSON object (no markdown formatting) with the following keys:\n"
        f"- 'verdict': One of ['Real', 'Fake', 'Misleading', 'Out-of-Context', 'Unverified']\n"
        f"- 'truthfulness_score': An integer from 0 to 100 (100 = Definitive Truth, 0 = Definitive Lie/Fake)\n"
        f"- 'explanation': A natural language paragraph explaining the reasoning.\n"
        f"- 'evidence': A list of bullet points citing specific inconsistencies or facts found via search.\n"
    )

    if image:
        prompt += "\nMedia is attached below.\n"

    # Request JSON response
    generation_config = {"response_mime_type": "application/json"}
    
    try:
        # Get analysis from Gemini
        response_text = get_gemini_response(
            prompt, 
            media_content=image, 
            model_name="gemini-flash-latest", 
            generation_config=generation_config
        )
        
        # Clean up code blocks if Gemini returns them despite being asked not to
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(cleaned_text)
        
        # Merge CLIP data into result for UI if needed
        result_json["clip_score"] = final_clip_score
        
        return result_json
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "verdict": "Error",
            "truthfulness_score": 0,
            "explanation": f"Failed to parse model response. Raw response: {response_text}",
            "evidence": []
        }
    except Exception as e:
        # General exception (e.g. API error)
        return {
            "verdict": "Error",
            "truthfulness_score": 0,
            "explanation": f"Analysis failed: {str(e)}",
            "evidence": []
        }
