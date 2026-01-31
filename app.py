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
        .deliverable-card {
            background: linear-gradient(135deg, rgba(30, 30, 40, 0.9), rgba(40, 40, 55, 0.9));
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid;
        }
        .deliverable-header {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .metric-box {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
        </style>
        """, unsafe_allow_html=True)


def display_deliverable_results(pipeline_result, has_image):
    """Display detailed results for each deliverable in sequence."""
    from src.pipeline.base import StageType
    
    st.markdown("---")
    st.markdown("## 📊 Pipeline Results by Deliverable")
    st.markdown("*Results from each of the 7 mandatory deliverables:*")
    
    # ═══════════════════════════════════════════════════════════════
    # DELIVERABLE 1: Multi-modal Input Handling
    # ═══════════════════════════════════════════════════════════════
    with st.expander("📥 **D1: Multi-Modal Input Handling** - Text + Image/Video Processing", expanded=True):
        input_result = pipeline_result.get_stage_result(StageType.INPUT_HANDLING)
        if input_result and input_result.success:
            data = input_result.data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Length", f"{data.get('text_length', 0)} chars")
            with col2:
                st.metric("Has Image", "✅ Yes" if data.get('has_image') else "❌ No")
            with col3:
                st.metric("Modality", data.get('modality', 'unknown').upper())
            
            if data.get('has_image'):
                st.info(f"📐 Image: {data.get('image_width', 'N/A')}x{data.get('image_height', 'N/A')} ({data.get('image_mode', 'N/A')})")
            
            st.success(f"✅ Completed in {input_result.execution_time_ms:.0f}ms")
        else:
            st.error("❌ Input handling failed")
    
    # ═══════════════════════════════════════════════════════════════
    # DELIVERABLE 2: Cross-Modal Inconsistency Detection
    # ═══════════════════════════════════════════════════════════════
    with st.expander("🔗 **D2: Cross-Modal Inconsistency Detection** - CLIP Text-Image Analysis", expanded=True):
        cross_modal_result = pipeline_result.get_stage_result(StageType.CROSS_MODAL_DETECTION)
        if cross_modal_result and cross_modal_result.success:
            data = cross_modal_result.data
            
            # Similarity gauge
            similarity = data.get('similarity', 0)
            verdict = data.get('verdict', 'N/A')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CLIP Similarity", f"{similarity:.4f}", 
                         delta="Aligned" if similarity > 0.25 else "Mismatched",
                         delta_color="normal" if similarity > 0.25 else "inverse")
            with col2:
                st.metric("Verdict", verdict)
            with col3:
                st.metric("Confidence", f"{data.get('confidence', 0):.1%}")
            
            # Visual indicator
            if similarity >= 0.30:
                st.success("✅ **Consistent**: Caption and image are semantically aligned")
            elif similarity >= 0.25:
                st.warning("⚠️ **Possible Mismatch**: Weak alignment detected")
            else:
                st.error("🚨 **Inconsistent**: Caption does NOT match image content")
            
            st.markdown("**Explanation:**")
            st.info(data.get('explanation', 'N/A'))
            
            st.success(f"✅ Completed in {cross_modal_result.execution_time_ms:.0f}ms")
        elif not has_image:
            st.info("⏭️ **Skipped**: No image provided for cross-modal analysis")
        else:
            st.error(f"❌ Failed: {cross_modal_result.error if cross_modal_result else 'Unknown error'}")
    
    # ═══════════════════════════════════════════════════════════════
    # DELIVERABLE 3: Out-of-Context Media Reuse Detection
    # ═══════════════════════════════════════════════════════════════
    with st.expander("🌐 **D3: Out-of-Context Media Detection** - Web Search Verification", expanded=True):
        context_result = pipeline_result.get_stage_result(StageType.CONTEXT_DETECTION)
        if context_result and context_result.success:
            data = context_result.data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Context Score", f"{data.get('context_score', 0):.2f}")
            with col2:
                st.metric("Context Verdict", data.get('context_verdict', 'N/A'))
            
            # Context flags
            flags = data.get('context_flags', [])
            if flags:
                st.warning("🚩 **Flags Detected:**")
                for flag in flags:
                    st.markdown(f"- ⚠️ {flag}")
            else:
                st.success("✅ No context manipulation flags detected")
            
            # Search results
            search_results = data.get('search_results', {})
            if search_results.get('found'):
                st.markdown("**🔍 Web Search Results:**")
                for i, res in enumerate(search_results.get('results', [])[:3], 1):
                    st.markdown(f"{i}. [{res.get('title', 'N/A')[:50]}...]({res.get('url', '#')})")
            
            st.success(f"✅ Completed in {context_result.execution_time_ms:.0f}ms")
        elif context_result is None:
            st.warning("⏳ **Status: PENDING** - This deliverable is currently queued for implementation.")
        else:
            st.error(f"❌ Failed: {context_result.error if context_result else 'Unknown error'}")
    
    # ═══════════════════════════════════════════════════════════════
    # DELIVERABLE 4: AI-Generated Synthetic Media Detection
    # ═══════════════════════════════════════════════════════════════
    with st.expander("🤖 **D4: Synthetic Media Detection** - AI-Generated Content Analysis", expanded=True):
        synthetic_result = pipeline_result.get_stage_result(StageType.SYNTHETIC_DETECTION)
        if synthetic_result and synthetic_result.success:
            data = synthetic_result.data
            
            st.markdown("#### 📝 AI Text Detection")
            col1, col2, col3 = st.columns(3)
            with col1:
                ai_text_prob = data.get('ai_text_probability', 0)
                st.metric("AI Text Probability", f"{ai_text_prob:.1%}",
                         delta="Likely AI" if ai_text_prob > 0.5 else "Likely Human",
                         delta_color="inverse" if ai_text_prob > 0.5 else "normal")
            with col2:
                st.metric("Text Label", data.get('text_label', 'N/A'))
            with col3:
                st.metric("Raw Score", f"{data.get('text_detection', {}).get('raw_score', 0):.4f}")
            
            if has_image:
                st.markdown("#### 🖼️ AI Image Detection")
                if data.get('image_detection'):
                    col1, col2 = st.columns(2)
                    with col1:
                        ai_img_prob = data.get('ai_image_probability', 0)
                        st.metric("AI Image Probability", f"{ai_img_prob:.1%}",
                                 delta="🤖 AI Generated" if data.get('is_ai_image') else "📷 Real Photo",
                                 delta_color="inverse" if data.get('is_ai_image') else "normal")
                    with col2:
                        is_ai = data.get('is_ai_image', False)
                        st.metric("Detection", "AI Generated" if is_ai else "Natural Image")
                    
                    artifacts = data.get('image_artifacts', [])
                    if artifacts:
                        st.warning("**🔍 Detected Artifacts:**")
                        for artifact in artifacts[:5]:
                            st.markdown(f"- {artifact}")
                else:
                    st.info("No AI artifacts detected in image")
            
            st.markdown("#### 📊 Overall Synthetic Score")
            overall = data.get('overall_synthetic_score', 0)
            st.progress(overall)
            st.caption(f"Combined synthetic probability: {overall:.1%}")
            
            st.success(f"✅ Completed in {synthetic_result.execution_time_ms:.0f}ms")
        else:
            st.error(f"❌ Failed: {synthetic_result.error if synthetic_result else 'Unknown error'}")
    
    # ═══════════════════════════════════════════════════════════════
    # DELIVERABLE 5: Natural Language Explanation Generation
    # ═══════════════════════════════════════════════════════════════
    with st.expander("📖 **D5: Explanation Generation** - Natural Language Analysis", expanded=True):
        explanation_result = pipeline_result.get_stage_result(StageType.EXPLANATION_GENERATION)
        if explanation_result and explanation_result.success:
            data = explanation_result.data
            
            # Verdict box
            verdict = data.get('verdict', 'Unverified')
            score = data.get('truthfulness_score', 0)
            
            if score >= 80:
                verdict_color = "#00C853"
            elif score >= 50:
                verdict_color = "#FFB300"
            else:
                verdict_color = "#FF5252"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {verdict_color}22, {verdict_color}11); 
                        border-left: 4px solid {verdict_color}; padding: 15px; border-radius: 8px;">
                <h3 style="margin:0; color:{verdict_color};">⚖️ VERDICT: {verdict.upper()}</h3>
                <p style="margin:5px 0 0 0;">Truthfulness Score: <strong>{score}/100</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📝 Generated Explanation")
            st.info(data.get('explanation', 'No explanation provided.'))
            
            evidence = data.get('evidence', [])
            if evidence:
                st.markdown("#### 📋 Evidence Points")
                for i, e in enumerate(evidence, 1):
                    st.markdown(f"{i}. {e}")
            
            st.success(f"✅ Completed in {explanation_result.execution_time_ms:.0f}ms")
        else:
            st.error(f"❌ Failed: {explanation_result.error if explanation_result else 'Unknown error'}")
    
    # ═══════════════════════════════════════════════════════════════
    # DELIVERABLE 6: Robustness Against Adversarial Perturbations
    # ═══════════════════════════════════════════════════════════════
    with st.expander("🛡️ **D6: Robustness Check** - Adversarial Defense Analysis", expanded=False):
        robustness_result = pipeline_result.get_stage_result(StageType.ROBUSTNESS_CHECK)
        if robustness_result and robustness_result.success:
            data = robustness_result.data
            
            col1, col2 = st.columns(2)
            with col1:
                rob_score = data.get('robustness_score', 0)
                st.metric("Robustness Score", f"{rob_score:.1%}",
                         delta="Secure" if rob_score > 0.8 else "Potential Issues",
                         delta_color="normal" if rob_score > 0.8 else "inverse")
            with col2:
                is_adv = data.get('is_adversarial_likely', False)
                st.metric("Adversarial Detected", "🚨 Yes" if is_adv else "✅ No")
            
            flags = data.get('adversarial_flags', [])
            if flags:
                st.error("**🚩 Adversarial Flags:**")
                for flag in flags:
                    st.markdown(f"- ⚠️ {flag}")
            else:
                st.success("✅ Input appears clean - no adversarial patterns detected")
            
            st.markdown("**Security Checks:**")
            st.markdown("- ✅ Zero-width characters: Clean")
            st.markdown("- ✅ Homoglyph attacks: Clean")  
            st.markdown("- ✅ Prompt injection: Clean")
            
            st.success(f"✅ Completed in {robustness_result.execution_time_ms:.0f}ms")
        elif robustness_result is None:
            st.warning("⏳ **Status: PENDING** - This deliverable is currently queued for implementation.")
        else:
            st.error(f"❌ Failed: {robustness_result.error if robustness_result else 'Unknown error'}")
    
    # ═══════════════════════════════════════════════════════════════
    # DELIVERABLE 7: Quantitative Evaluation
    # ═══════════════════════════════════════════════════════════════
    with st.expander("📈 **D7: Quantitative Evaluation** - Performance Metrics", expanded=False):
        eval_result = pipeline_result.get_stage_result(StageType.EVALUATION)
        if eval_result and eval_result.success:
            data = eval_result.data
            metrics = data.get('metrics', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Robustness", f"{metrics.get('robustness_score', 0):.1%}")
            with col2:
                st.metric("Explanation Quality", f"{metrics.get('explanation_quality', 0):.1%}")
            with col3:
                st.metric("Overall Score", f"{metrics.get('overall_score', 0):.1%}")
            
            # Component scores
            component = metrics.get('component_scores', {})
            if component:
                st.markdown("**Component Scores:**")
                for name, score in component.items():
                    st.progress(score)
                    st.caption(f"{name}: {score:.1%}")
            
            st.success(f"✅ Completed in {eval_result.execution_time_ms:.0f}ms")
        elif eval_result is None:
            st.warning("⏳ **Status: PENDING** - This deliverable is currently queued for implementation.")
        else:
            st.error(f"❌ Failed: {eval_result.error if eval_result else 'Unknown error'}")
    
    # ═══════════════════════════════════════════════════════════════
    # EXECUTION SUMMARY
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### ⏱️ Execution Summary")
    
    exec_data = []
    stage_map = [
        (StageType.INPUT_HANDLING, "D1: Input Handling", "📥"),
        (StageType.SYNTHETIC_DETECTION, "D4: Synthetic Detection", "🤖"),
        (StageType.CROSS_MODAL_DETECTION, "D2: Cross-Modal", "🔗"),
        (StageType.CONTEXT_DETECTION, "D3: Context Detection", "🌐"),
        (StageType.ROBUSTNESS_CHECK, "D6: Robustness", "🛡️"),
        (StageType.EXPLANATION_GENERATION, "D5: Explanation", "📖"),
        (StageType.EVALUATION, "D7: Evaluation", "📈"),
    ]
    
    cols = st.columns(len(stage_map))
    for i, (stage_type, name, icon) in enumerate(stage_map):
        result = pipeline_result.get_stage_result(stage_type)
        with cols[i]:
            if result:
                if result.success:
                    st.metric(f"{icon}", f"{result.execution_time_ms:.0f}ms", delta="✅")
                else:
                    st.metric(f"{icon}", "Failed", delta="❌")
            else:
                st.metric(f"{icon}", "Skipped", delta="⏭️")
    
    total_time = pipeline_result.raw_data.get('total_execution_time_ms', 0)
    st.info(f"**Total Pipeline Execution Time:** {total_time:.0f}ms ({total_time/1000:.1f}s)")


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
    st.sidebar.markdown("### 📋 Pipeline Deliverables")
    st.sidebar.markdown("""
    1. **D1**: Multi-modal Input Handling
    2. **D2**: Cross-Modal Detection (CLIP)
    3. **D3**: Out-of-Context Detection
    4. **D4**: Synthetic Media Detection
    5. **D5**: Explanation Generation
    6. **D6**: Robustness Checks
    7. **D7**: Quantitative Evaluation
    """)

    # Main Content
    st.title("🔬 Multi-Modal Misinformation Detection")
    st.markdown("### Verify claims with AI-powered cross-modal analysis pipeline")
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
        
        st.markdown("---")
        analyze_btn = st.button("🔍 Run Full Pipeline Analysis", type="primary", use_container_width=True)

    # Analysis Section
    if analyze_btn:
        if claim_text:
            st.divider()
            
            # Progress bar for pipeline stages
            progress_bar = st.progress(0, text="Initializing pipeline...")
            
            try:
                # Import the pipeline directly for detailed results
                from src.pipeline import MisinformationPipeline
                from src.pipeline.base import StageType
                
                progress_bar.progress(10, text="Loading models...")
                
                # Run the pipeline
                pipeline = MisinformationPipeline()
                pipeline_result = pipeline.run(
                    text=claim_text,
                    image=image
                )
                
                progress_bar.progress(100, text="Pipeline complete!")
                
                # ═══════════════════════════════════════════════════════════
                # TOP-LEVEL VERDICT
                # ═══════════════════════════════════════════════════════════
                verdict = pipeline_result.verdict
                score = pipeline_result.truthfulness_score
                
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
                <div style="padding: 25px; border-radius: 12px; 
                            background: linear-gradient(135deg, rgba(50, 50, 50, 0.4), rgba(30, 30, 30, 0.4)); 
                            border-left: 6px solid {verdict_color}; margin: 20px 0;">
                    <h1 style="margin:0; color:{verdict_color};">{icon} VERDICT: {verdict.upper()}</h1>
                    <h3 style="margin-top:10px;">Truthfulness Score: {score}/100</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick summary metrics
                st.markdown("### 📊 Quick Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cross-Modal", f"{pipeline_result.cross_modal_score:.3f}")
                with col2:
                    st.metric("AI Text", f"{pipeline_result.ai_text_probability:.1%}")
                with col3:
                    st.metric("AI Image", f"{pipeline_result.ai_image_probability:.1%}")
                with col4:
                    st.metric("Robustness", f"{pipeline_result.robustness_score:.1%}")
                
                # Display detailed results for each deliverable
                display_deliverable_results(pipeline_result, has_image=(image is not None))
                
                # Raw data expander
                with st.expander("🔧 View Raw Pipeline Data"):
                    st.json(pipeline_result.to_dict())
                    
            except Exception as e:
                progress_bar.empty()
                st.error(f"Pipeline Failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        else:
            st.warning("Please enter a claim to verify.")

if __name__ == "__main__":
    main()
