from src.models import configure_gemini, get_gemini_response
import json
from duckduckgo_search import DDGS
from src.clip_detector import detect_cross_modal
from src.synthetic_detector import detect_ai_text_pipeline, detect_ai_image_gemini
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
    image: Optional[Image.Image] = None
) -> Dict[str, Any]:
    """
    Analyzes the text and media to detect potential misinformation.
    
    Args:
        text (str): The claim or caption to analyze.
        media_path (str, optional): Path to image file.
        image (PIL.Image, optional): The image object.
        
    Returns:
        dict: A dictionary containing the structured analysis results.
    """
    # 0. Load Image (Unified logic)
    if image is None and media_path:
        try:
            image = Image.open(media_path).convert("RGB")
        except Exception:
            pass # Handle gracefully later

    # --- Common Step: AI Text Detection ---
    print("Running AI Text Detection...")
    ai_text_result = detect_ai_text_pipeline(text)
    
    # --- Common Step: Ensure Gemini is configured ---
    configure_gemini()
    
    # --- Common Step: Perform Search Grounding ---
    search_context = perform_verification_search(text)

    # Initialize variables for the result construction
    final_clip_score = 0.0
    ai_image_result = None
    prompt = ""

    # === BRANCH 1: MULTI-MODAL ANALYSIS (Text + Image) ===
    if image:
        print("Running AI Image Detection (Gemini)...")
        ai_image_result = detect_ai_image_gemini(image)

        # Run CLIP Cross-Modal Analysis
        clip_context = ""
        try:
            print("Running CLIP analysis...")
            clip_result = detect_cross_modal(text, image)
            final_clip_score = clip_result.similarity
            clip_context = (
                f"\nContext from CLIP Analysis:\n"
                f"- Semantic Similarity: {clip_result.similarity:.2f}\n"
                f"- Cross-Modal Verdict: {clip_result.verdict}\n"
                f"- Automatic Explanation: {clip_result.explanation}\n"
            )
        except Exception as e:
            print(f"CLIP analysis failed: {e}")
            clip_context = "\nCLIP analysis could not be performed.\n"
            
        # Construct Multi-Modal Prompt
        prompt = (
            f"You are an expert Multi-Modal Misinformation & AI Detection Assistant. "
            f"Your task is to analyze the provided claim and media to determine its truthfulness and origin.\n"
        )
        
        # Inject AI Detection Signals
        prompt += "\n--- SYSTEM DETECTION SIGNALS ---\n"
        prompt += f"AI Text Probability: {ai_text_result.get('ai_probability', 0):.2%} (Label: {ai_text_result.get('label')})\n"
        
        if ai_image_result:
            prompt += f"AI Image Probability: {ai_image_result.get('confidence_score', 0):.2%} (Is AI: {ai_image_result.get('is_ai_generated')})\n"
            if ai_image_result.get('artifacts'):
                 prompt += f"Detected Visual Artifacts: {', '.join(ai_image_result.get('artifacts', []))}\n"
        
        prompt += clip_context
        
        if final_clip_score < 0.20:
             prompt += " [(IMPORTANT) The CLIP score is VERY LOW. The image likely has nothing to do with the text.]"
                
        # Inject search results
        prompt += f"\n\nContext from Web Search (Verification Data):\n{search_context}\n"
        
        prompt += (
            f"\n\nClaim to Verify: {text}\n"
            f"Task Instructions:\n"
            f"1. **Origin Analysis**: Based on the System Detection Signals and your own analysis, explicitly state if the text or image is AI-generated. NOTE: A claim can be AI-generated but still Factually TRUE.\n"
            f"2. **Identity & Context Verification**: Use the Web Search Results to verify the identities, events, and facts. Check for hallucinations common in AI text.\n"
            f"3. **Cross-Modal Consistency**: Does the image support the claim? If it's an AI image, is it being passed off as real footage?\n"
            f"4. **Truthfulness Assessment**: Final verdict on the TRUTH of the claim. \n"
            f"   - If Text is AI-generated + Factually Correct -> Verdict: 'Real' (with note 'AI-Drafted').\n"
            f"   - If Image is AI-generated + Passed as Real -> Verdict: 'Fake' or 'Misleading'.\n"
            f"5. **Explanation**: Clear reasoning.\n"
        )
         
    # === BRANCH 2: TEXT-ONLY ANALYSIS ===
    else:
        # Construct Text-Only Prompt
        prompt = (
            f"You are an expert Fact-Checker and AI Text Detection Assistant. "
            f"Your task is to analyze the provided text to determine its truthfulness and origin.\n"
        )
        
        # Inject AI Detection Signals
        prompt += "\n--- SYSTEM DETECTION SIGNALS ---\n"
        prompt += f"AI Text Probability: {ai_text_result.get('ai_probability', 0):.2%} (Label: {ai_text_result.get('label')})\n"
        
        # Inject search results
        prompt += f"\n\nContext from Web Search (Verification Data):\n{search_context}\n"
        
        prompt += (
            f"\n\nClaim to Verify: {text}\n"
            f"Task Instructions:\n"
            f"1. **Origin Analysis**: Based on the AI Text Probability and wording, determine if the text is likely AI-generated. NOTE: A claim can be AI-generated but still Factually TRUE.\n"
            f"2. **Fact Verification**: Use the Web Search Results to verify the identities, events, and facts. Check for hallucinations common in AI text.\n"
            f"3. **Truthfulness Assessment**: Final verdict on the TRUTH of the claim. \n"
            f"   - If Text is AI-generated + Factually Correct -> Verdict: 'Real' (with note 'AI-Drafted').\n"
            f"   - If Text is Factually FALSE -> Verdict: 'Fake' or 'Misleading'.\n"
            f"4. **Explanation**: Clear reasoning focusing on factual accuracy.\n"
        )

    # --- Common Step: Final Output Formatting Instructions ---
    prompt += (
        f"\nOutput Format: Return the result as a raw JSON object (no markdown formatting) with the following keys:\n"
        f"- 'verdict': One of ['Real', 'Fake', 'Misleading', 'Out-of-Context', 'Unverified', 'AI-Generated (True)', 'AI-Generated (Fake)']\n"
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
            model_name="gemini-flash-lite-latest", 
            generation_config=generation_config
        )
        
        # Clean up code blocks if Gemini returns them despite being asked not to
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(cleaned_text)
        
        # Merge Technical Signals into result for UI
        result_json["ai_text_result"] = ai_text_result
        
        if image:
             result_json["clip_score"] = final_clip_score
             if ai_image_result:
                 result_json["ai_image_result"] = ai_image_result
        
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
