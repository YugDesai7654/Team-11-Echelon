import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def set_custom_style():
    st.markdown("""
        <style>
        .main {
            padding-top: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #FF2B2B;
            color: white;
            border-color: #FF2B2B;
        }
        h1 {
            color: #FAFAFA;
            font-family: 'Helvetica Neue', sans-serif;
        }
        h3 {
            color: #E0E0E0;
        }
        /* Improve JSON display readability */
        .e1jgr2j3 {
            background-color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Echelon | Truth & Misinformation Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_custom_style()

    # Sidebar
    st.sidebar.title("🛡️ Echelon")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        st.sidebar.success("Gemini API Key Detected ✅")
    else:
        st.sidebar.error("API Key Missing ❌")
        st.sidebar.info("Please set GOOGLE_API_KEY in .env")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**System Capabilities**\n\n"
        "1. **Analysis**: Checks for logical consistencies and manipulation.\n"
        "2. **Truthfulness**: Assigns a truth score based on world knowledge.\n"
        "3. **Context**: Detects if media is being used out-of-context.\n"
        "4. **CLIP**: Uses OpenAI CLIP to verify text-image grounding."
    )

    # Main Content
    st.title("Multi-Modal Misinformation Detection")
    st.markdown("### Verify claims with AI-powered cross-modal analysis.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.header("1. Upload Evidence")
        uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])
        
        image = None
        if uploaded_file is not None:
            file_type = uploaded_file.type.split('/')[0]
            if file_type == "image":
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Analyzed Media", use_container_width=True)
            elif file_type == "video":
                st.video(uploaded_file)
                st.info("Video upload successful. System will analyze context.")

    with col2:
        st.header("2. Claim to Verify")
        claim_text = st.text_area(
            "Enter the text/caption found with the media:",
            height=150,
            placeholder="Example: 'Breaking: Massive floods in Dubai today due to cloud seeding...'"
        )
        
        st.markdown("#### Options")
        use_search = st.checkbox("Enable Internet Search context (DuckDuckGo + Gemini)", value=True)
        
        st.markdown("---")
        analyze_btn = st.button("🔍 Verify Truthfulness", type="primary")

    # Analysis Section
    if analyze_btn:
        if claim_text:
            st.divider()
            st.subheader("Results")
            
            with st.spinner("Running Multi-Modal Analysis (CLIP + Search + Gemini)..."):
                try:
                    # Import here to avoid issues if not running
                    from src.analysis import detect_misinformation
                    
                    # Perform Analysis
                    # Note: detect_misinformation now handles CLIP internally if image is provided
                    result = detect_misinformation(
                        claim_text, 
                        image=image,
                        clip_score=0.0 # Will be recalculated by function
                    )
                except Exception as e:
                    st.error(f"Analysis Failed: {str(e)}")
                    result = {}

            if result:
                # Top Level Verdict
                verdict = result.get("verdict", "Unverified")
                score = result.get("truthfulness_score", 0)
                
                # Dynamic Color
                if score >= 80:
                    verdict_color = "green"
                    icon = "✅"
                elif score >= 50:
                    verdict_color = "orange"
                    icon = "⚠️"
                else:
                    verdict_color = "red"
                    icon = "🚨"

                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: rgba(50, 50, 50, 0.3); border-left: 5px solid {verdict_color};">
                    <h2 style="margin:0; color:{verdict_color};">{icon} Verdict: {verdict.upper()}</h2>
                    <h4 style="margin-top:5px;">Truthfulness Score: {score}/100</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Explanation
                st.markdown("### 📝 Detailed Explanation")
                st.write(result.get("explanation", "No explanation provided."))
                
                # Evidence Points
                if result.get("evidence"):
                    st.markdown("### 🔍 Key Evidence")
                    for item in result["evidence"]:
                        st.markdown(f"- {item}")
                
                # JSON/Debug Data
                with st.expander("View Raw Analysis Data"):
                    st.json(result)
                    if "clip_score" in result:
                        st.metric("Visual-Text Consistency (CLIP)", f"{result['clip_score']:.2f}")

        else:
            st.warning("Please enter a claim to verify.")

if __name__ == "__main__":
    main()
