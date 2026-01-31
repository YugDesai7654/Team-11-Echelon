import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

MODELS_TO_TEST = [
    "gemini-1.5-flash", 
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite-001"
]

print(f"Testing {len(MODELS_TO_TEST)} models for quota/access...")

for model_name in MODELS_TO_TEST:
    print(f"\n--- Testing: {model_name} ---")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, just checking connection.", request_options={"timeout": 10})
        print(f"✅ SUCCESS! Response: {response.text.strip()}")
        print(f"RECOMMENDATION: Use '{model_name}'")
        break
    except Exception as e:
        error_str = str(e)
        if "404" in error_str:
             print("❌ Not Found (404)")
        elif "429" in error_str:
             print("⚠️ Quota Exceeded (429)")
        else:
             print(f"❌ Error: {error_str[:100]}...")
    time.sleep(1) # Be nice
