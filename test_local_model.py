"""
Test script to check if we can use the Transformers model locally.
This tests downloading and using the model directly instead of via Inference API.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_local_model():
    """Test if we can load and use the model locally via transformers"""
    print("="*60)
    print("Testing Local Transformers Model")
    print("Model: prithivMLmods/Deep-Fake-Detector-v2-Model")
    print("="*60)
    
    try:
        print("\n✅ Importing transformers...")
        from transformers import pipeline
        from PIL import Image
        import numpy as np
        print("✅ Imports successful")
        
        print("\n📥 Loading model (this may take a few minutes on first run)...")
        # Use CPU (device=-1) to avoid CUDA issues
        pipe = pipeline(
            'image-classification',
            model="prithivMLmods/Deep-Fake-Detector-v2-Model",
            device=-1  # Use CPU
        )
        print("✅ Model loaded successfully!")
        
        print("\n🧪 Creating test image...")
        # Create a small test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        print("🔍 Running inference...")
        result = pipe(test_image)
        print(f"✅ Inference successful!")
        print(f"\nResult: {result}")
        
        print("\n" + "="*60)
        print("✅ SUCCESS: Model can be used locally!")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {str(e)}")
        print("   Make sure transformers, torch, and pillow are installed")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        print(f"\nFull traceback:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    test_local_model()
