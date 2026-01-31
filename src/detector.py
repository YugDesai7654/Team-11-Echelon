import torch
from transformers import pipeline
from PIL import Image

class AdvancedMisinfoDetector:
    def __init__(self):
        # We use a generative multi-modal model for reasoning
        # Note: 'llava-hf/llava-1.5-7b-hf' or 'LanguageBind/Video-LLaVA-7B-hf' are great choices
        self.pipe = pipeline("visual-question-answering", model="llava-hf/llava-1.5-7b-hf")

    def analyze_with_reasoning(self, image_path, claim):
        image = Image.open(image_path)
        
        # We ask the model a specific "Verification" question
        prompt = f"USER: <image>\nThe claim is: '{claim}'. Does this image provide evidence for or against this claim? Explain why accurately. \nASSISTANT:"
        
        # The model generates a textual explanation (Deliverable 5)
        outputs = self.pipe(image, prompt, generate_kwargs={"max_new_tokens": 200})
        return outputs[0]['answer']