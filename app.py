import streamlit as st
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = "AIzaSyCs2bCsgMfXssG3-O5dWJH5Nl4UfjIAoR4"
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="Multi-Modal Misinfo Detector", layout="wide")

st.title("🛡️ AI Misinformation Detection System")
st.markdown("Detecting cross-modal inconsistencies using Gemini.")

st.sidebar.header("Upload Section")
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
claim_text = st.sidebar.text_area("Enter Claim to Verify", "e.g., This video shows a protest in Surat in 2026.")

with st.sidebar.expander("🔍 Available Models"):
    try:
        models = genai.list_models()
        st.write("**Models with generateContent support:**")
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name
                available_models.append(model_name)
                st.write(f"- {model_name}")
        
        if not available_models:
            st.warning("No models found with generateContent support.")
    except Exception as e:
        st.error(f"Error listing models: {e}")

if uploaded_file and st.sidebar.button("Analyze Video"):
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Video")
        st.video(temp_path)

    with col2:
        st.subheader("AI Verification Report")
        with st.spinner("Uploading and analyzing video... This may take a minute."):
            try:
                video_file = genai.upload_file(path=temp_path)
                
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)

                prompt = f"""
                Analyze this video specifically for misinformation regarding the claim: '{claim_text}'
                1. Check for cross-modal inconsistencies (Does the video match the text?).
                2. Identify if the media looks AI-generated (Deepfake) or out-of-context.
                3. Provide a natural language explanation citing specific visual/audio evidence.
                4. Give a final 'Confidence Score' from 0 to 100.
                """

                models = genai.list_models()
                model_found = False
                response = None
                
                for model in models:
                    if 'generateContent' in model.supported_generation_methods:
                        model_name = model.name
                        try:
                            gen_model = genai.GenerativeModel(model_name=model_name)
                            response = gen_model.generate_content([video_file, prompt])
                            model_found = True
                            st.success(f"✅ Using model: {model_name}")
                            break
                        except Exception as e:
                            st.warning(f"⚠️ Model {model_name} failed: {str(e)[:100]}")
                            continue
                
                if model_found and response:
                    st.markdown("### AI Verdict")
                    st.write(response.text)
                    st.success("Analysis Complete!")
                else:
                    st.error("❌ All available models failed. Please check:")
                    st.write("1. Your API key has access to video-capable models")
                    st.write("2. The video file is in a supported format")
                    st.write("3. Check the 'Available Models' section in the sidebar")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                with st.expander("Full Error Details"):
                    st.code(traceback.format_exc())
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)