"""
Multi-Modal Misinformation Detection - Streamlit App

Simple CLIP-based detection of caption-image consistency.
"""

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from src.analysis import detect_misinformation


def get_verdict_color(verdict: str) -> str:
    """Get color based on verdict."""
    if "Consistent" in verdict or "authentic" in verdict.lower():
        return "green"
    elif "Inconsistent" in verdict or "misinformation" in verdict.lower():
        return "red"
    else:
        return "orange"


def display_result(result: dict):
    """Display detection results."""
    verdict = result["verdict"]
    confidence = result["confidence"]
    similarity = result.get("similarity", 0)
    
    # Main verdict with color
    color = get_verdict_color(verdict)
    st.markdown(f"### Verdict: :{color}[{verdict}]")
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", f"{confidence:.1%}")
    with col2:
        st.metric("CLIP Similarity", f"{similarity:.4f}")
    
    st.divider()
    
    # Explanation
    st.subheader("📝 Explanation")
    st.markdown(result["explanation"])
    
    # Evidence
    if result.get("evidence"):
        st.subheader("📊 Evidence")
        st.info(result["evidence"])


def main():
    st.set_page_config(
        page_title="Multi-Modal Misinformation Detector",
        page_icon="🔍",
        layout="wide",
    )
    
    # Header
    st.title("🔍 Multi-Modal Misinformation Detection")
    st.caption("CLIP-based analysis of text + image consistency")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input")
        caption = st.text_area(
            "Caption / Claim",
            placeholder="Enter the text that accompanies the image...",
            height=120,
        )
        uploaded = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "webp"],
        )
    
    with col2:
        st.subheader("🖼️ Preview")
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)
        else:
            st.info("Upload an image to preview")
    
    st.divider()
    
    # Run button
    if st.button("🔍 Run Detection", type="primary"):
        if not caption or not caption.strip():
            st.warning("⚠️ Please enter a caption or claim.")
        elif uploaded is None:
            st.warning("⚠️ Please upload an image.")
        else:
            with st.spinner("🔄 Running CLIP analysis..."):
                img = Image.open(uploaded).convert("RGB")
                result = detect_misinformation(
                    text=caption.strip(),
                    image=img,
                )
            
            if result["verdict"] == "Error":
                st.error(f"❌ Error: {result['explanation']}")
            else:
                display_result(result)
    
    # Footer
    st.divider()
    st.caption("CLIP-based misinformation detection. Always verify important claims with additional sources.")


if __name__ == "__main__":
    main()
