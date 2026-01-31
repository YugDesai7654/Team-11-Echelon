import torch
import numpy as np
import clip
import cv2
from decord import VideoReader, cpu
from PIL import Image
from transformers import XCLIPProcessor, XCLIPModel

model_name = "microsoft/xclip-base-patch16-zero-shot"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

def analyze_video_consistency(video_path, claim_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    vr = VideoReader(video_path, ctx=cpu(0))
    indices = np.linspace(0, len(vr) - 1, num=8).astype(int)
    video_data = vr.get_batch(indices).asnumpy()
    
    if len(video_data) == 0:
        raise ValueError("No frames extracted from video")
    
    video_frames_list = [Image.fromarray(frame.astype(np.uint8)) for frame in video_data]
    
    try:
        pixel_values = processor.image_processor(
            videos=video_frames_list,
            return_tensors="pt"
        )["pixel_values"]
        
        text_inputs = processor.tokenizer(
            [claim_text],
            return_tensors="pt",
            padding=True
        )
        
        inputs = {
            "pixel_values": pixel_values.to(device),
            "input_ids": text_inputs["input_ids"].to(device),
            "attention_mask": text_inputs["attention_mask"].to(device)
        }
    except Exception as e:
        raise ValueError(f"Processor error: {str(e)}")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_video.softmax(dim=-1).cpu().numpy()
        score = float(probs[0][0]) 
        
    return score, video_data