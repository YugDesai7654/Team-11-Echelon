"""
Multi-Modal Misinformation Detection with Explanation Generation
=================================================================
A Streamlit application for detecting AI-generated synthetic media and 
generating human-understandable explanations.

This application addresses the following Mandatory Deliverables:
- #1: Multi-modal input handling (text + video)
- #4: Detection of AI-generated synthetic media (deepfakes)
- #5: Natural language explanation generation citing concrete evidence
"""

import streamlit as st
import cv2
import numpy as np
import requests
import tempfile
import os
import base64
import logging
from io import BytesIO
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# API Keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face API endpoint for fake image detection
# Using Vision Transformer (ViT) based deepfake detector - state-of-the-art as of 2026
HF_API_URL = "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

# ============================================================================
# DELIVERABLE #1: Multi-modal input handling (text + image and/or video)
# ============================================================================
def save_uploaded_video(uploaded_file):
    """
    Saves the uploaded video to a temporary file for processing.
    
    ADDRESSES DELIVERABLE #1: Handles video input from the user.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        str: Path to the saved temporary file
    """
    # Create a temporary file to store the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(uploaded_file.read())
    temp_file.close()
    return temp_file.name


def extract_frames(video_path, num_frames=3):
    """
    Extracts Start, Middle, and End frames from the video using OpenCV.
    
    ADDRESSES DELIVERABLE #1: Processes video input by extracting key frames
    for analysis.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (default: 3)
        
    Returns:
        list: List of extracted frames as numpy arrays (BGR format)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return frames
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 3:
        st.warning("Video has fewer than 3 frames. Extracting all available frames.")
        num_frames = total_frames
    
    # Calculate frame indices: Start (0), Middle (50%), End (last)
    if num_frames == 3:
        frame_indices = [0, total_frames // 2, total_frames - 1]
    else:
        frame_indices = list(range(total_frames))
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


# ============================================================================
# DELIVERABLE #4: Detection of AI-generated synthetic media (deepfakes)
# ============================================================================
def frame_to_base64(frame):
    """
    Converts an OpenCV frame (numpy array) to base64 encoded JPEG.
    
    Args:
        frame: OpenCV frame in BGR format
        
    Returns:
        bytes: JPEG encoded image bytes
    """
    # Convert BGR to RGB for proper color representation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame_rgb)
    return buffer.tobytes()


def analyze_frame_for_deepfake(frame_bytes, max_retries=3):
    """
    Sends a frame to Hugging Face API for deepfake/AI-generated image detection.
    
    ADDRESSES DELIVERABLE #4: Uses AI model to detect synthetic media.
    
    Args:
        frame_bytes: JPEG encoded image bytes
        max_retries: Maximum number of retries if model is loading
        
    Returns:
        float or None: Fake probability (0-1) or None if API fails
    """
    import time
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/octet-stream"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL, 
                headers=headers, 
                data=frame_bytes,
                timeout=60  # 60 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # Parse the response to find the "artificial" or "fake" probability
                for item in result:
                    label = item.get("label", "").lower()
                    if "artificial" in label or "fake" in label or "ai" in label:
                        return item.get("score", 0)
                # If no explicit fake label, return 1 - real probability
                for item in result:
                    label = item.get("label", "").lower()
                    if "human" in label or "real" in label:
                        return 1 - item.get("score", 1)
                return 0.5  # Default if labels not recognized
            elif response.status_code == 503:
                # Model is loading - wait and retry
                try:
                    error_data = response.json()
                    wait_time = error_data.get("estimated_time", 20)
                except:
                    wait_time = 20
                if attempt < max_retries - 1:
                    st.info(f"🔄 Model is loading... waiting {wait_time:.0f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(min(wait_time, 30))  # Cap wait time at 30 seconds
                    continue
                return None
            else:
                st.warning(f"⚠️ Hugging Face API returned status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"⚠️ Timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                continue
            st.warning("⚠️ Hugging Face API timeout after all retries.")
            return None
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Hugging Face API error: {str(e)}")
            return None
    
    return None


def calculate_deepfake_score(frames):
    """
    Analyzes all frames and calculates the average deepfake probability.
    
    ADDRESSES DELIVERABLE #4: Provides quantitative deepfake detection score.
    
    Args:
        frames: List of OpenCV frames
        
    Returns:
        tuple: (average_score or "Unknown", list of individual scores)
    """
    scores = []
    
    with st.spinner("🔍 Analyzing frames for AI-generated content..."):
        progress_bar = st.progress(0)
        
        for i, frame in enumerate(frames):
            frame_bytes = frame_to_base64(frame)
            score = analyze_frame_for_deepfake(frame_bytes)
            scores.append(score)
            progress_bar.progress((i + 1) / len(frames))
    
    # Filter out None values
    valid_scores = [s for s in scores if s is not None]
    
    if not valid_scores:
        return "Unknown", scores
    
    average_score = sum(valid_scores) / len(valid_scores)
    return average_score, scores


# ============================================================================
# DELIVERABLE #5: Natural language explanation generation citing concrete evidence
# ============================================================================
def generate_explanation_with_gemini(text_claim, deepfake_score, frame_scores):
    """
    Uses Google Gemini to generate a human-understandable explanation
    of whether the video supports the text claim or is misinformation.
    
    ADDRESSES DELIVERABLE #5: Generates natural language explanations
    citing concrete evidence from the analysis.
    
    Args:
        text_claim: The user's text claim about the video
        deepfake_score: Average deepfake probability (0-1 or "Unknown")
        frame_scores: List of individual frame scores
        
    Returns:
        tuple: (verdict, explanation)
    """
    try:
        from google import genai
        
        logger.info("Initializing Gemini client...")
        # Initialize the client with API key
        client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("Gemini client initialized successfully")
        
        # Format the score information
        if deepfake_score == "Unknown":
            score_info = "The deepfake detection system was unable to analyze the video (API unavailable)."
            score_percentage = "Unknown"
        else:
            score_percentage = f"{deepfake_score * 100:.1f}%"
            frame_score_str = ", ".join([
                f"{s*100:.1f}%" if s is not None else "Unknown" 
                for s in frame_scores
            ])
            score_info = f"""
            - Average AI-Generated Probability: {score_percentage}
            - Frame-by-frame scores: {frame_score_str}
            - A score above 50% suggests the video may contain AI-generated or manipulated content.
            """
        
        prompt = f"""
        You are an expert forensic analyst specializing in detecting misinformation and deepfakes.
        
        TASK: Analyze the following evidence and determine if the video supports the text claim or if it's potentially misinformation.
        
        TEXT CLAIM: "{text_claim}"
        
        DEEPFAKE ANALYSIS RESULTS:
        {score_info}
        
        Based on this evidence, provide:
        1. A VERDICT: Either "LIKELY AUTHENTIC", "POTENTIALLY MANIPULATED", "LIKELY MISINFORMATION", or "INCONCLUSIVE"
        2. A detailed EXPLANATION (2-3 paragraphs) that:
           - Cites the specific deepfake score as evidence
           - Explains what the score means for the credibility of the video
           - Discusses whether the video can reliably support the text claim
           - Provides recommendations for the user
        
        Format your response as:
        VERDICT: [Your verdict]
        
        EXPLANATION:
        [Your detailed explanation]
        """
        
        logger.info("Sending request to Gemini API with model: models/gemini-1.5-flash")
        response = client.models.generate_content(
            model="models/gemini-1.5-flash",
            contents=[prompt]
        )
        response_text = response.text
        logger.info("Successfully received response from Gemini API")
        logger.debug(f"Response preview: {response_text[:200]}...")
        
        # Parse the response
        if "VERDICT:" in response_text and "EXPLANATION:" in response_text:
            parts = response_text.split("EXPLANATION:")
            verdict_part = parts[0].replace("VERDICT:", "").strip()
            explanation_part = parts[1].strip() if len(parts) > 1 else ""
            return verdict_part, explanation_part
        else:
            return "ANALYSIS COMPLETE", response_text
            
    except Exception as e:
        logger.error(f"Gemini API Error: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return "ERROR", f"Failed to generate explanation: {str(e)}"




# ============================================================================
# UI/UX: Dashboard Display
# ============================================================================
def display_dashboard(frames, deepfake_score, frame_scores, verdict, explanation):
    """
    Displays the analysis results in a comprehensive dashboard.
    
    ADDRESSES DELIVERABLE #5: Presents evidence and explanations in a
    human-understandable format.
    """
    st.markdown("---")
    st.header("📊 Analysis Dashboard")
    
    # Display the 3 extracted frames
    st.subheader("🎬 Extracted Evidence Frames")
    cols = st.columns(3)
    frame_labels = ["Start Frame", "Middle Frame", "End Frame"]
    
    for i, (col, frame, label) in enumerate(zip(cols, frames, frame_labels)):
        with col:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=label, width='stretch')
            if frame_scores[i] is not None:
                st.caption(f"AI Score: {frame_scores[i]*100:.1f}%")
            else:
                st.caption("AI Score: Unknown")
    
    # Display Deepfake Score
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Deepfake Detection Score")
        if deepfake_score == "Unknown":
            st.metric(
                label="AI-Generated Probability",
                value="Unknown",
                help="The detection API was unavailable"
            )
            st.warning("⚠️ Could not determine deepfake probability. Please try again later.")
        else:
            percentage = deepfake_score * 100
            st.metric(
                label="AI-Generated Probability",
                value=f"{percentage:.1f}%",
                delta=None
            )
            
            # Color-coded progress bar
            if percentage < 30:
                st.success(f"🟢 Low risk of AI manipulation ({percentage:.1f}%)")
            elif percentage < 70:
                st.warning(f"🟡 Moderate risk of AI manipulation ({percentage:.1f}%)")
            else:
                st.error(f"🔴 High risk of AI manipulation ({percentage:.1f}%)")
            
            st.progress(deepfake_score)
    
    with col2:
        st.subheader("⚖️ Final Verdict")
        
        # Style the verdict based on content
        if "AUTHENTIC" in verdict.upper():
            st.success(f"✅ {verdict}")
        elif "MISINFORMATION" in verdict.upper() or "MANIPULATED" in verdict.upper():
            st.error(f"🚨 {verdict}")
        elif "INCONCLUSIVE" in verdict.upper():
            st.warning(f"❓ {verdict}")
        else:
            st.info(f"📋 {verdict}")
    
    # Display detailed explanation
    st.markdown("---")
    st.subheader("📝 Detailed Explanation")
    st.markdown(explanation)
    
    # Evidence summary
    st.markdown("---")
    st.subheader("📋 Evidence Summary")
    evidence_data = {
        "Frame": frame_labels,
        "AI-Generated Score": [
            f"{s*100:.1f}%" if s is not None else "Unknown" 
            for s in frame_scores
        ]
    }
    st.table(evidence_data)


# ============================================================================
# Main Application
# ============================================================================
def main():
    """
    Main application entry point.
    
    ADDRESSES DELIVERABLE #1: Provides multi-modal input interface for
    text claims and video uploads.
    """
    # Page configuration
    st.set_page_config(
        page_title="Multi-Modal Misinformation Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("🔍 Multi-Modal Misinformation Detection")
    st.markdown("""
    ### AI-Powered Forensic Analysis Tool
    
    This tool analyzes videos for signs of AI-generated or manipulated content (deepfakes) 
    and evaluates whether the video supports a given text claim.
    
    **Mandatory Deliverables Addressed:**
    - ✅ **#1**: Multi-modal input handling (text + video)
    - ✅ **#4**: Detection of AI-generated synthetic media (deepfakes)
    - ✅ **#5**: Natural language explanation generation citing concrete evidence
    """)
    
    # Log API configuration at startup
    logger.info("="*60)
    logger.info("🚀 STREAMLIT APP STARTED")
    logger.info("="*60)
    logger.info(f"📍 Working Directory: {os.getcwd()}")
    
    # Check for API keys
    if not GOOGLE_API_KEY:
        logger.error("❌ GOOGLE_API_KEY not found!")
        st.error("⚠️ GOOGLE_API_KEY not found in .env file!")
        st.stop()
    else:
        masked_key = f"{GOOGLE_API_KEY[:8]}...{GOOGLE_API_KEY[-4:]}" if len(GOOGLE_API_KEY) > 12 else "***"
        logger.info(f"✅ Google Gemini API Key: {masked_key}")
        logger.info(f"🤖 Using Model: models/gemini-1.5-flash")
        logger.info(f"🔗 API Endpoint: https://generativelanguage.googleapis.com/")
    
    if not HUGGINGFACE_API_KEY:
        logger.error("❌ HUGGINGFACE_API_KEY not found!")
        st.error("⚠️ HUGGINGFACE_API_KEY not found in .env file!")
        st.stop()
    else:
        masked_hf_key = f"{HUGGINGFACE_API_KEY[:8]}...{HUGGINGFACE_API_KEY[-4:]}" if len(HUGGINGFACE_API_KEY) > 12 else "***"
        logger.info(f"✅ Hugging Face API Key: {masked_hf_key}")
        logger.info(f"🔬 Deepfake Model: prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-based)")
        logger.info(f"🔗 HF API Endpoint: {HF_API_URL}")
    
    logger.info("="*60)
    
    st.markdown("---")
    
    # Input Section (DELIVERABLE #1: Multi-modal input handling)
    st.header("📤 Input Section")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎥 Upload Video")
        uploaded_video = st.file_uploader(
            "Upload an MP4 video for analysis",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_video:
            st.video(uploaded_video)
    
    with col2:
        st.subheader("📝 Enter Text Claim")
        text_claim = st.text_area(
            "What claim does this video supposedly support?",
            placeholder="e.g., 'This video shows the president making a controversial statement about policy X.'",
            height=150,
            help="Enter the claim or context associated with this video"
        )
    
    # Analysis Button
    st.markdown("---")
    
    if st.button("🚀 Analyze for Misinformation", type="primary"):
        if not uploaded_video:
            st.error("Please upload a video file.")
        elif not text_claim.strip():
            st.error("Please enter a text claim.")
        else:
            # Process the video
            with st.spinner("Processing video..."):
                # Save video temporarily
                video_path = save_uploaded_video(uploaded_video)
                
                # Extract frames (DELIVERABLE #1)
                st.info("📹 Extracting frames from video...")
                frames = extract_frames(video_path)
                
                if len(frames) < 3:
                    st.error("Could not extract enough frames from the video.")
                    os.unlink(video_path)  # Clean up
                    st.stop()
                
                # Analyze for deepfakes (DELIVERABLE #4)
                st.info("🔬 Running deepfake detection analysis...")
                deepfake_score, frame_scores = calculate_deepfake_score(frames)
                
                # Generate explanation with Gemini (DELIVERABLE #5)
                st.info("🧠 Generating AI-powered explanation...")
                verdict, explanation = generate_explanation_with_gemini(
                    text_claim, deepfake_score, frame_scores
                )
                
                # Display dashboard
                display_dashboard(frames, deepfake_score, frame_scores, verdict, explanation)
                
                # Clean up temporary file
                os.unlink(video_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🔬 Multi-Modal Misinformation Detection Tool | Hackathon Project</p>
        <p>Powered by OpenCV, Hugging Face, and Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
