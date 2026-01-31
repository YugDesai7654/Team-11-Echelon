import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="Multi-Modal Misinformation Detector", layout="wide")
    st.title("Multi-Modal Misinformation Detection")
    st.sidebar.title("Navigation")
    
    st.write("Welcome to the Misinformation Detection System. Please upload media and text to verify.")

if __name__ == "__main__":
    main()
