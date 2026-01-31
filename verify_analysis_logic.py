
import sys
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# Mock the modules before importing src.analysis
sys.modules['src.models'] = MagicMock()
sys.modules['duckduckgo_search'] = MagicMock()
sys.modules['src.clip_detector'] = MagicMock()
sys.modules['src.synthetic_detector'] = MagicMock()

# Now import the function to test
from src.analysis import detect_misinformation

class TestAnalysisLogic(unittest.TestCase):
    
    @patch('src.analysis.detect_ai_text_pipeline')
    @patch('src.analysis.detect_ai_image_gemini')
    @patch('src.analysis.detect_cross_modal')
    @patch('src.analysis.get_gemini_response')
    @patch('src.analysis.perform_verification_search')
    def test_text_only_flow(self, mock_search, mock_gemini, mock_clip, mock_img_ai, mock_text_ai):
        # Setup Mocks
        mock_text_ai.return_value = {"ai_probability": 0.1, "label": "Human"}
        mock_search.return_value = "Search results"
        mock_gemini.return_value = '{"verdict": "Real", "truthfulness_score": 90, "explanation": "Test", "evidence": []}'
        
        # Call function
        result = detect_misinformation("Test claim")
        
        # Verify Interactions
        mock_text_ai.assert_called_once()
        mock_search.assert_called_once()
        
        # Start CRITICAL CHECKS
        mock_img_ai.assert_not_called() # Should NOT run image AI
        mock_clip.assert_not_called() # Should NOT run CLIP
        
        # Check Prompt Construction (Indirectly via Gemini call)
        args, _ = mock_gemini.call_args
        prompt = args[0]
        self.assertIn("expert Fact-Checker", prompt)
        self.assertNotIn("CLIP", prompt)
        self.assertNotIn("Media is attached below", prompt)
        
        # Check Result Logic
        self.assertNotIn("clip_score", result)
        self.assertNotIn("ai_image_result", result)
        print("✅ Text-Only Flow Verified")

    @patch('src.analysis.detect_ai_text_pipeline')
    @patch('src.analysis.detect_ai_image_gemini')
    @patch('src.analysis.detect_cross_modal')
    @patch('src.analysis.get_gemini_response')
    @patch('src.analysis.perform_verification_search')
    def test_multimodal_flow(self, mock_search, mock_gemini, mock_clip, mock_img_ai, mock_text_ai):
        # Setup Mocks
        mock_text_ai.return_value = {"ai_probability": 0.1, "label": "Human"}
        mock_img_ai.return_value = {"confidence_score": 0.9, "is_ai_generated": True}
        
        clip_mock_obj = MagicMock()
        clip_mock_obj.similarity = 0.85
        clip_mock_obj.verdict = "Match"
        clip_mock_obj.explanation = "Good match"
        mock_clip.return_value = clip_mock_obj
        
        mock_search.return_value = "Search results"
        mock_gemini.return_value = '{"verdict": "Fake", "truthfulness_score": 10, "explanation": "Test", "evidence": []}'
        
        # Create Mock Image
        img = Image.new('RGB', (100, 100))
        
        # Call function
        result = detect_misinformation("Test claim", image=img)
        
        # Verify Interactions
        mock_text_ai.assert_called_once()
        mock_img_ai.assert_called_once()
        mock_clip.assert_called_once()
        
        # Check Prompt Construction
        args, _ = mock_gemini.call_args
        prompt = args[0]
        self.assertIn("Multi-Modal Misinformation", prompt)
        self.assertIn("Context from CLIP Analysis", prompt)
        
        # Check Result Logic
        self.assertIn("clip_score", result)
        self.assertEqual(result["clip_score"], 0.85)
        self.assertIn("ai_image_result", result)
        print("✅ Multi-Modal Flow Verified")

if __name__ == '__main__':
    unittest.main()
