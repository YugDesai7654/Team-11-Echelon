from src.models import load_clip_model, configure_gemini, get_gemini_response
import json
from duckduckgo_search import DDGS

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

def detect_misinformation(text, media_content=None, clip_score=None):
    """
    Analyzes the text and media to detect potential misinformation, using Gemini 
    and incorporating CLIP similarity context.
    
    Args:
        text (str): The claim or caption to analyze.
        media_content (PIL.Image or list, optional): The image(s) associated with the claim.
        clip_score (float, optional): The semantic similarity score from CLIP (0.0 to 1.0).
        
    Returns:
        dict: A dictionary containing the structured analysis results.
    """
    # Ensure Gemini is configured
    configure_gemini()
    
    # perform search grounding
    search_context = perform_verification_search(text)
    
    # Construct the prompt with CLIP context and specific analysis instructions
    prompt = (
        f"You are an expert Multi-Modal Misinformation Detection AI. "
        f"Your task is to analyze the provided claim and media to determine its truthfulness. "
    )
    
    if clip_score is not None:
        prompt += f"\nContext: A CLIP model analyzed the semantic similarity between the text and image and gave a score of {clip_score:.2f} (on a scale of 0.0 to 1.0). "
        if clip_score < 0.25:
            prompt += "This low score suggests the text and image might be unrelated or mismatched. "
        elif clip_score > 0.3:
            prompt += "This score suggests some semantic relevant, but check for subtle manipulation. "
            
    # Inject search results into prompt
    prompt += f"\n\nContext from Web Search (Verification Data):\n{search_context}\n"
    
    prompt += (
        f"\n\nClaim to Verify: {text}\n"
        f"Task Instructions:\n"
        f"1. **Visual Analysis**: detailedly describe what is happening in the image/video.\n"
        f"2. **Identity & Context Verification (CRITICAL)**: Use the provided Web Search Results to verify the identities of any people in the image/claim. Check for public profiles (LinkedIn, Instagram, etc.) to confirm if the person is a local resident or public figure as claimed.\n"
        f"3. **Cross-Modal Consistency**: Does the image actually support the specific details in the claim? (e.g., location, time, people, events).\n"
        f"4. **Truthfulness Assessment**: Based on your knowledge and the provided Search Results, is this claim true? If it refers to a real event, is the image from that event or is it out-of-context (reused)?\n"
        f"5. **Explanation**: Provide a clear, step-by-step reasoning for your verdict, citing the search results where applicable.\n"
        f"\nOutput Format: Return the result as a raw JSON object (no markdown formatting) with the following keys:\n"
        f"- 'verdict': One of ['Real', 'Fake', 'Misleading', 'Out-of-Context', 'Unverified']\n"
        f"- 'truthfulness_score': An integer from 0 to 100 (100 = Definitive Truth, 0 = Definitive Lie/Fake)\n"
        f"- 'explanation': A natural language paragraph explaining the reasoning.\n"
        f"- 'evidence': A list of bullet points citing specific inconsistencies or facts found via search.\n"
    )

    if media_content:
        prompt += "\nMedia is attached below.\n"

    # Request JSON response
    generation_config = {"response_mime_type": "application/json"}
    
    try:
        # Get analysis from Gemini
        # Note: We manually injected search context, so we don't need 'tools' param here.
        # We use flash-latest as it is the most reliable available model.
        response_text = get_gemini_response(
            prompt, 
            media_content=media_content, 
            model_name="gemini-flash-latest", 
            generation_config=generation_config
        )
        
        # Clean up code blocks if Gemini returns them despite being asked not to
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(cleaned_text)
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
