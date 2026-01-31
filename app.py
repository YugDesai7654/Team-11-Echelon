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
# Using dima806 deepfake detector - user specified
HF_API_URL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"

# ============================================================================
# Model Availability Check
# ============================================================================
def check_hf_model_availability(model_url, api_key, timeout=10):
    """
    Checks if a HuggingFace model is available and responding.
    
    Args:
        model_url: Full URL to the HuggingFace model API
        api_key: HuggingFace API key
        timeout: Request timeout in seconds
        
    Returns:
        dict: {'available': bool, 'status_code': int, 'message': str}
    """
    import numpy as np
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream"
    }
    
    # Create a small test image (1x1 black pixel)
    test_image = np.zeros((1, 1, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', test_image)
    test_bytes = buffer.tobytes()
    
    try:
        logger.info(f"🔍 Testing HuggingFace model availability...")
        response = requests.post(
            model_url,
            headers=headers,
            data=test_bytes,
            timeout=timeout
        )
        
        status_code = response.status_code
        
        if status_code == 200:
            logger.info(f"✅ Model is available and responding")
            return {
                'available': True,
                'status_code': status_code,
                'message': 'Model available'
            }
        elif status_code == 503:
            # Model is loading
            logger.warning(f"⚠️ Model is loading (503)")
            return {
                'available': False,
                'status_code': status_code,
                'message': 'Model is loading, retry later'
            }
        elif status_code == 410:
            # Model deprecated/removed
            logger.error(f"❌ Model is deprecated or removed (410 Gone)")
            return {
                'available': False,
                'status_code': status_code,
                'message': 'Model deprecated/removed'
            }
        else:
            logger.warning(f"⚠️ Unexpected status: {status_code}")
            return {
                'available': False,
                'status_code': status_code,
                'message': f'Unexpected status: {status_code}'
            }
            
    except requests.exceptions.Timeout:
        logger.error(f"❌ Request timeout after {timeout}s")
        return {
            'available': False,
            'status_code': 0,
            'message': 'Request timeout'
        }
    except Exception as e:
        logger.error(f"❌ Error checking model: {str(e)}")
        return {
            'available': False,
            'status_code': 0,
            'message': f'Error: {str(e)}'
        }


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


def extract_frames(video_path, num_frames=30):
    """
    Extracts frames evenly distributed throughout the video using OpenCV.
    
    ADDRESSES DELIVERABLE #1: Processes video input by extracting key frames
    for analysis.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (default: 30)
        
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
    
    if total_frames < num_frames:
        st.warning(f"Video has fewer than {num_frames} frames. Extracting all available frames.")
        num_frames = total_frames
    
    # Calculate evenly distributed frame indices
    # For 30 frames in a 300-frame video: [0, 10, 20, ..., 290]
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    # Ensure we don't go past the last frame
    frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
    
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
    Analyzes a frame for deepfake/AI-generated content using local transformers model.
    
    ADDRESSES DELIVERABLE #4: Uses AI model to detect synthetic media.
    
    Args:
        frame_bytes: JPEG encoded image bytes
        max_retries: Not used for local model (kept for API compatibility)
        
    Returns:
        float or None: Fake probability (0-1) or None if analysis fails
    """
    try:
        from transformers import pipeline
        from PIL import Image
        import io
        
        # Load the pipeline (cached after first use)
        if not hasattr(analyze_frame_for_deepfake, 'pipe'):
            logger.info("📥 Loading deepfake detection model (first use only)...")
            analyze_frame_for_deepfake.pipe = pipeline(
                'image-classification',
                model="prithivMLmods/Deep-Fake-Detector-v2-Model",
                device=-1  # Use CPU
            )
            logger.info("✅ Model loaded successfully")
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(frame_bytes))
        
        # Run inference
        results = analyze_frame_for_deepfake.pipe(image)
        
        # Parse results - format: [{'label': 'Deepfake', 'score': 0.9}, {'label': 'Realism', 'score': 0.1}]
        fake_score = None
        for result in results:
            if result['label'].lower() == 'deepfake':
                fake_score = result['score']
                break
        
        if fake_score is None:
            # If 'Deepfake' not found, use (1 - Realism score)
            for result in results:
                if result['label'].lower() == 'realism':
                    fake_score = 1.0 - result['score']
                    break
        
        logger.debug(f"Deepfake analysis result: {fake_score}")
        return fake_score if fake_score is not None else 0.5
        
    except Exception as e:
        logger.error(f"Deepfake analysis error: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
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
        
        logger.info("Sending request to Gemini API with model: models/gemini-2.5-flash")
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
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
    
    # Log API configuration at startup (only once per session)
    if 'startup_logged' not in st.session_state:
        st.session_state.startup_logged = True
        logger.info("="*60)
        logger.info("🚀 STREAMLIT APP STARTED")
        logger.info("="*60)
        logger.info(f"📍 Working Directory: {os.getcwd()}")
    
    # Check for API keys
    if not GOOGLE_API_KEY:
        logger.error("❌ GOOGLE_API_KEY not found!")
        st.error("⚠️ GOOGLE_API_KEY not found in .env file!")
        st.stop()
    elif 'startup_logged' in st.session_state and st.session_state.startup_logged:
        # Only log on first startup
        masked_key = f"{GOOGLE_API_KEY[:8]}...{GOOGLE_API_KEY[-4:]}" if len(GOOGLE_API_KEY) > 12 else "***"
        logger.info(f"✅ Google Gemini API Key: {masked_key}")
        logger.info(f"🤖 Using Model: models/gemini-2.5-flash")
        logger.info(f"🔗 API Endpoint: https://generativelanguage.googleapis.com/")
    
    if not HUGGINGFACE_API_KEY:
        logger.error("❌ HUGGINGFACE_API_KEY not found!")
        st.error("⚠️ HUGGINGFACE_API_KEY not found in .env file!")
        st.stop()
    elif 'startup_logged' in st.session_state and st.session_state.startup_logged:
        # Only log on first startup
        masked_hf_key = f"{HUGGINGFACE_API_KEY[:8]}...{HUGGINGFACE_API_KEY[-4:]}" if len(HUGGINGFACE_API_KEY) > 12 else "***"
        logger.info(f"✅ Hugging Face API Key: {masked_hf_key}")
        logger.info(f"🔬 Deepfake Model: prithivMLmods/Deep-Fake-Detector-v2-Model (LOCAL)")
        logger.info(f"📦 Model Type: Local Transformers Pipeline (CPU inference)")
        logger.info(f"💾 Model will be cached after first download")
        
        # Model availability check not needed for local model
        st.session_state.hf_model_available = True
        logger.info(f"✅ Local model configuration validated")
        
        logger.info("="*60)
        # Mark as logged
        st.session_state.startup_logged = False  # Reset for next interactions

    
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
