# Multi-Modal Misinformation Detection with Explanation Generation

## Problem Statement
Misinformation increasingly spreads through multi-modal content—a mix of text, images, and videos—making detection significantly harder. A single modality may appear legitimate, while cross-modal inconsistencies reveal manipulation.
The task is to build a multi-modal AI system that detects misinformation and generates human-understandable explanations, while remaining robust against adversarial attempts and unseen manipulation techniques.

## Features
- **Multi-modal Input Handling**: Supports Text, Image, and Video inputs.
- **Cross-Modal Inconsistency Detection**: Uses CLIP to analyze the relationship between caption and media.
- **Explanation Generation**: Uses Google Gemini to provide natural language explanations for the verdict.
- **Synthetic Media Detection**: (Planned) Identification of AI-generated content.
- **Robustness**: Designed to attempt to withstand adversarial perturbations.

## Tech Stack
-   **Frontend**: Streamlit
-   **Backend**: Python
-   **Models**: CLIP (OpenAI/HuggingFace), Google Gemini

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd Team11-Echelon
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**:
    Create a `.env` file in the root directory and add your API keys:
    ```
    GOOGLE_API_KEY=your_gemini_api_key
    ```

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
