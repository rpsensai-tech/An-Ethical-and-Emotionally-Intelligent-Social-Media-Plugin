"""
Sarcasm Detection Service
Detects sarcasm in text using transformer-based models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional
import logging
from pathlib import Path


class SarcasmDetectionService:
    """
    Service for detecting sarcasm in text.
    Uses fine-tuned RoBERTa model for sarcasm classification.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the sarcasm detection service.
        
        Args:
            model_path: Path to fine-tuned sarcasm model
            device: Computing device (cuda/cpu/auto)
        """
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Initializing SarcasmDetectionService on {self.device}")
        
        # Model paths
        if model_path is None:
            # Default to sarcasm-detector in models directory
            model_path = Path(__file__).parent.parent.parent / "models" / "text" / "sarcasm-detector"
        
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        
        # Load model if path exists
        if self.model_path.exists():
            self._load_model()
        else:
            self.logger.warning(f"Sarcasm model not found at {self.model_path}")
    
    def _load_model(self):
        """Load the sarcasm detection model."""
        try:
            self.logger.info(f"Loading sarcasm model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("✅ Sarcasm model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load sarcasm model: {e}")
            raise
    
    def detect(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Detect sarcasm in a single text.
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for sarcasm detection
            
        Returns:
            Dictionary with sarcasm detection results
        """
        if self.model is None:
            raise RuntimeError("Sarcasm model not loaded. Cannot perform detection.")
        
        try:
            # Tokenize
            encodings = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**encodings)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            
            # Extract probabilities
            non_sarcasm_prob = float(probabilities[0][0].item())
            sarcasm_prob = float(probabilities[0][1].item())
            is_sarcastic = sarcasm_prob >= threshold
            
            return {
                'text': text,
                'is_sarcastic': is_sarcastic,
                'sarcasm_detected': is_sarcastic,
                'confidence': float(probabilities[0][predictions[0]].item()),
                'sarcasm_probability': sarcasm_prob,
                'non_sarcasm_probability': non_sarcasm_prob,
                'threshold': threshold
            }
            
        except Exception as e:
            self.logger.error(f"Sarcasm detection failed: {e}")
            raise RuntimeError(f"Failed to detect sarcasm: {e}") from e
    
    def detect_batch(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Detect sarcasm in multiple texts efficiently.
        
        Args:
            texts: List of input texts
            threshold: Confidence threshold
            
        Returns:
            List of detection results
        """
        if self.model is None:
            raise RuntimeError("Sarcasm model not loaded. Cannot perform detection.")
        
        try:
            # Tokenize batch
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict batch
            with torch.no_grad():
                outputs = self.model(**encodings)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            
            # Format results
            results = []
            for i, text in enumerate(texts):
                sarcasm_prob = float(probabilities[i][1].item())
                is_sarcastic = sarcasm_prob >= threshold
                
                results.append({
                    'text': text,
                    'is_sarcastic': is_sarcastic,
                    'sarcasm_detected': is_sarcastic,
                    'confidence': float(probabilities[i][predictions[i]].item()),
                    'sarcasm_probability': sarcasm_prob,
                    'non_sarcasm_probability': float(probabilities[i][0].item()),
                    'threshold': threshold
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch sarcasm detection failed: {e}")
            raise RuntimeError(f"Failed to detect sarcasm in batch: {e}") from e
    
    def adjust_emotion_for_sarcasm(self, emotion: str, is_sarcastic: bool) -> str:
        """
        Adjust emotion prediction based on sarcasm detection.
        Sarcasm typically reverses emotional polarity.
        
        Args:
            emotion: Original emotion prediction
            is_sarcastic: Whether sarcasm was detected
            
        Returns:
            Adjusted emotion
        """
        if not is_sarcastic:
            return emotion
        
        # Emotion polarity reversal mapping
        reversal_map = {
            'joy': 'sadness',
            'happy': 'sad',
            'sadness': 'joy',
            'sad': 'happy',
            'love': 'disgust',
            'admiration': 'disapproval',
            'approval': 'disapproval',
            'disapproval': 'approval',
            'excitement': 'disappointment',
            'gratitude': 'annoyance',
            'pride': 'embarrassment',
            'relief': 'nervousness',
            'amusement': 'boredom',
            'desire': 'disgust',
            # Neutral emotions typically don't reverse
            'neutral': 'neutral',
            'surprise': 'surprise',
            'realization': 'confusion',
            'confusion': 'realization',
        }
        
        emotion_lower = emotion.lower()
        adjusted = reversal_map.get(emotion_lower, emotion)
        
        # Preserve original casing
        if emotion[0].isupper():
            adjusted = adjusted.capitalize()
        
        return adjusted
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded sarcasm model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "model_type": "RoBERTa-based sarcasm detector",
            "capabilities": ["binary_classification", "batch_processing", "emotion_adjustment"]
        }
