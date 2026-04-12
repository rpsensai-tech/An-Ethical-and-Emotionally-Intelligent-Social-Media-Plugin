"""
Text Prediction Service
Handles text-based emotion detection and prediction with sarcasm and slang detection
"""

import torch
from typing import Dict, List, Any, Optional
import time
import logging
from pathlib import Path
import sys

from assets.configs.emotion_config import emotion_config

from components.emotion.app.model_config.text_emotion_classifier import TextModelManager, TextEmotionClassifier
from ..utils.preprocessing import TextPreprocessor, EmotionMapper
from ..utils.device_manager import device_manager
from transformers import AutoTokenizer
from .enhanced_text_service import EnhancedTextAnalysisService


class TextPredictionService:
    """Handles text emotion prediction with sarcasm and slang detection"""
    
    def __init__(self, models_dir: Path, emotions: List[str], device: str = "auto"):
        self.models_dir = Path(models_dir)
        self.emotions = emotions
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_manager = TextModelManager(models_dir, emotions)
        self.device = device_manager.get_device(device)
        self.preprocessor = None
        
        # Initialize enhanced text analysis (sarcasm + slang)
        try:
            self.enhanced_service = EnhancedTextAnalysisService(device=str(self.device))
            self.logger.info("Enhanced text analysis (sarcasm + slang) enabled")
        except Exception as e:
            self.logger.warning(f"Enhanced text analysis not available: {e}")
            self.enhanced_service = None
        
        self.logger.info(f"TextPredictionService initialized on {self.device}")
    
    def load_model(self, model_name: str = "default") -> bool:
        """Load a text emotion model"""
        try:
            model, tokenizer = self.model_manager.load_model(model_name, self.device)
            
            self.model_manager.current_model = model
            self.model_manager.current_tokenizer = tokenizer
            self.model_manager.current_device = self.device
            self.model_manager.current_model_name = model_name
            
            # Initialize preprocessor with same tokenizer
            self.preprocessor = TextPreprocessor(
                model_name=model.model_name,
                max_length=128
            )
            
            self.logger.info(f"✅ Loaded text model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load text model: {e}")
            return False
    
    def predict_emotions(self, text: str, threshold: float = 0.3, 
                        include_sarcasm: bool = True, 
                        include_slang: bool = True) -> Dict[str, Any]:
        """
        Predict emotions from text with optional sarcasm and slang detection
        
        Args:
            text: Input text to analyze
            threshold: Minimum probability threshold for emotion detection
            include_sarcasm: Include sarcasm detection analysis
            include_slang: Include slang detection analysis
        
        Returns:
            Dictionary with emotion predictions and metadata
        """
        
        start_time = time.time()
        
        try:
            # Auto-load default model if none is loaded
            if self.model_manager.current_model is None:
                self.logger.info("No model loaded, loading 'default' model...")
                self.model_manager.load_model("default", self.device)
            
            if self.model_manager.current_model is None:
                raise RuntimeError("No model loaded. Please load a model first.")
            
            # Make prediction
            result = self.model_manager.predict_with_current_model(text, threshold=threshold)
            
            # Get emotion probabilities
            emotion_probs = result['emotion_probabilities']
            
            # Filter significant emotions
            significant_emotions = {
                emotion: prob for emotion, prob in emotion_probs.items()
                if prob >= threshold
            }
            
            # Get top emotions
            sorted_emotions = sorted(
                emotion_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_emotion = sorted_emotions[0][0]
            top_probability = sorted_emotions[0][1]
            
            # Map to core emotions if using GoEmotions
            core_emotions = emotion_config.map_to_core_emotion(emotion_probs)
            
            # Get sentiment
            sentiment = emotion_config.get_sentiment(emotion_probs, threshold)
            
            # Format results
            emotion_list = []
            for emotion, probability in sorted_emotions:
                confidence_level = self._get_confidence_level(probability)
                emotion_list.append({
                    'emotion': emotion,
                    'probability': round(probability, 4),
                    'confidence_level': confidence_level
                })
            
            processing_time = time.time() - start_time
            
            # Base result
            base_result = {
                'text': text,
                'emotions': emotion_list,
                'significant_emotions': {k: round(v, 4) for k, v in significant_emotions.items()},
                'top_emotion': top_emotion,
                'top_probability': round(top_probability, 4),
                'core_emotions': {k: round(v, 4) for k, v in core_emotions.items()},
                'sentiment': sentiment,
                'processing_time': round(processing_time, 4),
                'model_used': self.model_manager.current_model_name or "default",
                'threshold': threshold,
                'modality': 'text'
            }
            
            # Add enhanced analysis (sarcasm + slang) if enabled
            if (include_sarcasm or include_slang) and self.enhanced_service:
                try:
                    enhanced_analysis = self.enhanced_service.analyze_comprehensive(
                        text=text,
                        emotion_result=base_result
                    )
                    
                    # Add enhanced data to result
                    if include_sarcasm and enhanced_analysis.get('sarcasm'):
                        base_result['sarcasm'] = enhanced_analysis['sarcasm']
                    
                    if include_slang and enhanced_analysis.get('slang'):
                        base_result['slang'] = enhanced_analysis['slang']
                    
                    # Add recommendations
                    if enhanced_analysis.get('recommendations'):
                        base_result['recommendations'] = enhanced_analysis['recommendations']
                    
                    # Add emotion adjustment suggestion if sarcasm detected
                    if enhanced_analysis.get('emotion_adjustment'):
                        base_result['emotion_adjustment'] = enhanced_analysis['emotion_adjustment']
                    
                except Exception as e:
                    self.logger.warning(f"Enhanced analysis failed: {e}")
            
            return base_result
            
        except Exception as e:
            self.logger.error(f"Text prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def predict_batch(self, texts: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Predict emotions for multiple texts"""
        results = []
        for text in texts:
            try:
                result = self.predict_emotions(text, threshold)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch prediction failed for text: {e}")
                results.append({
                    'text': text,
                    'error': str(e)
                })
        return results
    
    def _get_confidence_level(self, probability: float) -> str:
        """Map probability to confidence level"""
        if probability >= 0.8:
            return "very_high"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        elif probability >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """List available text models"""
        return self.model_manager.list_models()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model"""
        if self.model_manager.current_model is None:
            return {"status": "no_model_loaded"}
        
        return {
            "model_name": self.model_manager.current_model_name,
            "model_info": self.model_manager.current_model.get_model_info(),
            "device": str(self.device),
            "emotions": self.emotions
        }
    
    def explain_prediction(self, text: str, model_name: str = "default", 
                          method: str = "attention") -> Dict[str, Any]:
        """
        Generate explanation for emotion prediction
        
        Args:
            text: Input text to explain
            model_name: Model to use for prediction
            method: Explanation method (attention, shap, lime)
        
        Returns:
            Dictionary with explanation data
        """
        try:
            # Ensure model is loaded
            if self.model_manager.current_model is None or \
               (model_name != "default" and model_name != self.model_manager.current_model_name):
                self.load_model(model_name)
            
            # Get prediction first
            prediction = self.predict_emotions(text, threshold=0.1)
            top_emotion = prediction['top_emotion']
            
            # Get top emotion index
            try:
                top_emotion_idx = self.emotions.index(top_emotion)
            except ValueError:
                top_emotion_idx = 0
            
            # Initialize explainer
            from ..model_config.explainer import TextModelExplainer
            explainer = TextModelExplainer(
                self.model_manager.current_model,
                self.model_manager.current_tokenizer,
                self.device,
                self.emotions
            )
            
            # Map 'attention' to 'lime' as fallback since attention is not implemented
            if method == "attention":
                method = "lime"
                self.logger.info("Using LIME as fallback for attention-based explanation")
            
            # Generate explanation based on method - use reduced samples for faster response
            if method == "shap":
                self.logger.info(f"Generating SHAP explanation for emotion: {top_emotion}")
                explanation = explainer.explain_with_shap(text, top_emotion_idx=top_emotion_idx, max_evals=50)
            elif method == "lime":
                self.logger.info(f"Generating LIME explanation for emotion: {top_emotion}")
                # Use reduced samples for faster response (100 instead of 1000)
                explanation = explainer.explain_with_lime(text, top_emotion_idx=top_emotion_idx, num_samples=100)
            else:
                raise ValueError(f"Unknown explanation method: {method}. Supported: shap, lime")
            
            self.logger.info(f"Explanation generated successfully using method: {explanation.get('method', method)}")
            
            return {
                'text': text,
                'top_emotion': top_emotion,
                'top_probability': prediction['top_probability'],
                'explanation': explanation,
                'method': explanation.get('method', method),
                'model_used': self.model_manager.current_model_name
            }
            
        except Exception as e:
            self.logger.error(f"Explanation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate explanation: {e}") from e
