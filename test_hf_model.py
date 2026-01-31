"""
Test script to check if a HuggingFace model is available.
Usage: python test_hf_model.py
"""

import os
import cv2
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_hf_model_availability(model_url, api_key, timeout=10):
    """
    Checks if a HuggingFace model is available and responding.
    
    Args:
        model_url: Full URL to the HuggingFace model API
        api_key: HuggingFace API key
        timeout: Request timeout in seconds
        
    Returns:
        dict: {'available': bool, 'status_code': int, 'message': str}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream"
    }
    
    # Create a small test image (1x1 black pixel)
    test_image = np.zeros((1, 1, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', test_image)
    test_bytes = buffer.tobytes()
    
    try:
        print(f"🔍 Testing HuggingFace model...")
        print(f"   URL: {model_url}")
        
        response = requests.post(
            model_url,
            headers=headers,
            data=test_bytes,
            timeout=timeout
        )
        
        status_code = response.status_code
        
        if status_code == 200:
            print(f"✅ Model is available and responding (200 OK)")
            try:
                data = response.json()
                print(f"   Response: {data}")
            except:
                pass
            return {
                'available': True,
                'status_code': status_code,
                'message': 'Model available'
            }
        elif status_code == 503:
            print(f"⚠️  Model is loading (503 Service Unavailable)")
            return {
                'available': False,
                'status_code': status_code,
                'message': 'Model is loading, retry later'
            }
        elif status_code == 410:
            print(f"❌ Model is deprecated or removed (410 Gone)")
            return {
                'available': False,
                'status_code': status_code,
                'message': 'Model deprecated/removed'
            }
        else:
            print(f"⚠️  Unexpected status: {status_code}")
            return {
                'available': False,
                'status_code': status_code,
                'message': f'Unexpected status: {status_code}'
            }
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout after {timeout}s")
        return {
            'available': False,
            'status_code': 0,
            'message': 'Request timeout'
        }
    except Exception as e:
        print(f"❌ Error checking model: {str(e)}")
        return {
            'available': False,
            'status_code': 0,
            'message': f'Error: {str(e)}'
        }


if __name__ == "__main__":
    # Test the requested model
    HF_API_URL = "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-v2-Model"
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    if not HUGGINGFACE_API_KEY:
        print("❌ HUGGINGFACE_API_KEY not found in .env file!")
    else:
        print("="*60)
        print("HuggingFace Model Availability Test")
        print("Model: prithivMLmods/Deep-Fake-Detector-v2-Model")
        print("="*60)
        result = check_hf_model_availability(HF_API_URL, HUGGINGFACE_API_KEY)
        print("\nFinal Result:")
        print(f"  Available: {result['available']}")
        print(f"  Status Code: {result['status_code']}")
        print(f"  Message: {result['message']}")
        print("="*60)
