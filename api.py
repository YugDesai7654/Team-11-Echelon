
import os
import shutil
import tempfile
import uvicorn
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Import pipeline
from src.pipeline import MisinformationPipeline
from src.utils.chat_utils import get_chat_response

app = FastAPI(title="Echelon API", description="Backend for Misinformation Detection")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline once
pipeline = MisinformationPipeline()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    pipeline_result: Dict

@app.get("/")
def health_check():
    return {"status": "online", "service": "Echelon API"}

@app.post("/analyze")
async def analyze_media(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Analyze text + media (image/video).
    """
    try:
        content_type = file.content_type
        print(f"Received file: {file.filename} ({content_type})")
        
        image = None
        video_path = None
        temp_file_path = None

        # Handle Image
        if content_type.startswith("image/"):
            file_bytes = await file.read()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # Handle Video
        elif content_type.startswith("video/"):
            suffix = os.path.splitext(file.filename)[1]
            if not suffix:
                suffix = ".mp4"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                video_path = tmp.name
                temp_file_path = tmp.name
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image or video.")

        # Run pipeline
        print("Running pipeline...")
        result = pipeline.run(
            text=text,
            image=image,
            video_path=video_path
        )
        
        # Cleanup temp video file if exists
        # Note: In a real app, you might want to keep it longer or clean up in background task
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return result.to_dict()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_evidence(req: ChatRequest):
    try:
        response = get_chat_response(
            messages=req.messages,
            pipeline_result=req.pipeline_result
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
