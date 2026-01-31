# Multi-Modal Misinformation Detection

CLIP-based detection of caption-image consistency for identifying potential misinformation.

## Features

- **Cross-Modal Analysis**: Uses CLIP to measure text-image alignment
- **Similarity Score**: 0-1 score showing how well caption matches image
- **Verdict**: Consistent / Possible misinformation / Inconsistent
- **Simple Explanation**: Human-readable assessment

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## How It Works

1. Upload an image
2. Enter the caption/claim
3. Click "Run Detection"
4. Get CLIP similarity score and verdict

## Project Structure

```
├── app.py              # Streamlit web app
├── src/
│   ├── analysis.py     # Main detection pipeline
│   ├── clip_detector.py # CLIP cross-modal detection
│   └── models.py       # CLIP model loading
├── requirements.txt    # Dependencies
└── README.md
```

## Tech Stack

- **CLIP**: OpenAI's CLIP model for cross-modal understanding
- **Streamlit**: Web interface
- **PyTorch**: Deep learning backend
- **Transformers**: HuggingFace model loading
