"""
Model Explainability Module
Provides SHAP and LIME explanations for emotion predictions
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from transformers import AutoTokenizer


class TextModelExplainer:
    """Provides SHAP and LIME explanations for text emotion predictions"""
    
    def __init__(self, model, tokenizer: AutoTokenizer, 
                 device: torch.device, emotions: List[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.emotions = emotions
        self.logger = logging.getLogger(__name__)
        
        # Verify model output size matches emotions
        self.num_emotions = len(emotions)
        if hasattr(model, 'num_emotions'):
            if model.num_emotions != self.num_emotions:
                self.logger.warning(f"Model has {model.num_emotions} emotions but {self.num_emotions} emotion labels provided")
                self.num_emotions = min(model.num_emotions, self.num_emotions)
        
        # Initialize explainers
        self.shap_explainer = None
        if LIME_AVAILABLE:
            self.lime_explainer = LimeTextExplainer(class_names=emotions[:self.num_emotions])
            self.logger.info(f"LIME explainer initialized with {self.num_emotions} emotions")
        else:
            self.lime_explainer = None
            self.logger.warning("LIME not available. Install with: pip install lime")
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def _predict_function(self, texts: List[str]) -> np.ndarray:
        """Prediction function for LIME/SHAP"""
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                # Handle empty text
                if not text or not text.strip():
                    # Return neutral probabilities for empty text
                    predictions.append(np.ones(len(self.emotions)) / len(self.emotions))
                    continue
                    
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                logits = self.model(inputs['input_ids'], inputs['attention_mask'])
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
                predictions.append(probabilities)
        
        result = np.array(predictions)
        self.logger.debug(f"Prediction shape: {result.shape} for {len(texts)} texts")
        return result
    
    def explain_with_lime(self, text: str, top_emotion_idx: int, 
                         num_features: int = 10, num_samples: int = 100) -> Dict[str, Any]:
        """Generate LIME explanation"""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return self._get_fallback_explanation(text, top_emotion_idx, 'lime')
        
        try:
            self.logger.info(f"Generating LIME explanation for emotion index {top_emotion_idx} ({self.emotions[top_emotion_idx]})")
            
            # Validate emotion index
            if top_emotion_idx >= len(self.emotions):
                self.logger.error(f"Invalid emotion index {top_emotion_idx}, max is {len(self.emotions)-1}")
                return self._get_fallback_explanation(text, top_emotion_idx, 'lime')
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                text, 
                self._predict_function,
                labels=[top_emotion_idx],
                num_features=num_features,
                num_samples=num_samples
            )
            
            self.logger.info("LIME explanation generated successfully")
            
            # Extract feature importance
            features = []
            try:
                feature_list = explanation.as_list(label=top_emotion_idx)
                for word, importance in feature_list:
                    features.append({
                        'word': word,
                        'importance': float(importance),
                        'position': text.lower().find(word.lower()) if word.lower() in text.lower() else -1
                    })
            except Exception as e:
                self.logger.error(f"Failed to extract features from LIME explanation: {e}")
                return self._get_fallback_explanation(text, top_emotion_idx, 'lime')
            
            # Sort by absolute importance
            features.sort(key=lambda x: abs(x['importance']), reverse=True)
            
            # Extract confidence safely
            confidence = 0.0
            if hasattr(explanation, 'local_pred'):
                try:
                    # For multi-label, local_pred might be 1D array with predictions for requested label only
                    if len(explanation.local_pred) == 1:
                        confidence = float(explanation.local_pred[0])
                    elif top_emotion_idx < len(explanation.local_pred):
                        confidence = float(explanation.local_pred[top_emotion_idx])
                except (IndexError, TypeError) as e:
                    self.logger.warning(f"Could not extract confidence from LIME: {e}")
            
            return {
                'method': 'lime',
                'emotion': self.emotions[top_emotion_idx],
                'features': features[:num_features],
                'explanation_summary': self._generate_explanation_summary(features[:5], 'LIME'),
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"LIME explanation failed: {e}", exc_info=True)
            return self._get_fallback_explanation(text, top_emotion_idx, 'lime')
    
    def explain_with_shap(self, text: str, top_emotion_idx: int, 
                         max_evals: int = 100) -> Dict[str, Any]:
        """Generate SHAP explanation"""
        if not SHAP_AVAILABLE:
            return self._get_fallback_explanation(text, top_emotion_idx, 'shap')
        
        try:
            # Initialize SHAP explainer if not already done
            if self.shap_explainer is None:
                # Create a simple background dataset
                background_texts = [
                    "This is neutral text.",
                    "I am happy today.",
                    "I am sad about this.",
                    "This makes me angry.",
                    "",  # Empty text
                ]
                self.shap_explainer = shap.Explainer(
                    self._predict_function, 
                    background_texts,
                    max_evals=max_evals
                )
            
            # Generate SHAP values
            shap_values = self.shap_explainer([text])
            
            # Extract SHAP values for the top emotion
            emotion_shap_values = shap_values[:, :, top_emotion_idx]
            
            # Get words and their importance
            words = text.split()
            if len(words) != emotion_shap_values.shape[1]:
                # Fallback to word-level tokenization
                words = self.tokenizer.tokenize(text)
                if len(words) > emotion_shap_values.shape[1]:
                    words = words[:emotion_shap_values.shape[1]]
                elif len(words) < emotion_shap_values.shape[1]:
                    words.extend(['[PAD]'] * (emotion_shap_values.shape[1] - len(words)))
            
            features = []
            for i, (word, importance) in enumerate(zip(words, emotion_shap_values[0])):
                if word not in ['[PAD]', '[CLS]', '[SEP]']:
                    features.append({
                        'word': word,
                        'importance': float(importance),
                        'position': i
                    })
            
            # Sort by absolute importance
            features.sort(key=lambda x: abs(x['importance']), reverse=True)
            
            return {
                'method': 'shap',
                'emotion': self.emotions[top_emotion_idx],
                'features': features[:10],
                'explanation_summary': self._generate_explanation_summary(features[:5], 'SHAP'),
                'base_value': float(shap_values.base_values[0][top_emotion_idx]) if hasattr(shap_values, 'base_values') else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"SHAP explanation failed: {e}")
            return self._get_fallback_explanation(text, top_emotion_idx, 'shap')
    
    def _generate_explanation_summary(self, top_features: List[Dict], method: str) -> str:
        """Generate human-readable explanation summary"""
        if not top_features:
            return f"{method} analysis could not identify significant features."
        
        positive_words = [f['word'] for f in top_features if f['importance'] > 0]
        negative_words = [f['word'] for f in top_features if f['importance'] < 0]
        
        summary_parts = []
        
        if positive_words:
            summary_parts.append(f"Words supporting this emotion: {', '.join(positive_words[:3])}")
        
        if negative_words:
            summary_parts.append(f"Words opposing this emotion: {', '.join(negative_words[:3])}")
        
        if not summary_parts:
            summary_parts.append("No clear linguistic indicators found.")
        
        return f"{method} explanation: " + "; ".join(summary_parts)
    
    def _get_fallback_explanation(self, text: str, top_emotion_idx: int, method: str) -> Dict[str, Any]:
        """Provide fallback explanation when SHAP/LIME fails"""
        words = text.split()
        
        # Simple keyword-based explanation
        emotion_keywords = {
            'joy': ['happy', 'great', 'awesome', 'love', 'amazing', 'wonderful', 'excited'],
            'sadness': ['sad', 'terrible', 'awful', 'bad', 'disappointed', 'upset', 'depressed'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'annoying', 'stupid', 'rage'],
            'fear': ['scared', 'afraid', 'worried', 'nervous', 'anxious', 'terrified'],
            'surprise': ['wow', 'amazing', 'incredible', 'unbelievable', 'shocking', 'surprised'],
            'disgust': ['gross', 'disgusting', 'terrible', 'awful', 'nasty', 'repulsive'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'regular'],
            'happy': ['happy', 'great', 'awesome', 'love', 'amazing', 'wonderful']
        }
        
        emotion_name = self.emotions[top_emotion_idx] if top_emotion_idx < len(self.emotions) else 'unknown'
        keywords = emotion_keywords.get(emotion_name, [])
        
        features = []
        for word in words:
            if word.lower() in keywords:
                features.append({
                    'word': word,
                    'importance': 0.5,  # Default importance
                    'position': text.lower().find(word.lower())
                })
        
        return {
            'method': f'{method}_fallback',
            'emotion': emotion_name,
            'features': features[:5],
            'explanation_summary': f"Fallback analysis identified potential {emotion_name}-related words.",
            'note': f"Original {method} analysis unavailable, using keyword-based fallback."
        }
    
    def explain_prediction(self, text: str, method: str = 'lime') -> Dict[str, Any]:
        """Generate explanation for a text prediction"""
        # First get prediction to find top emotion
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        top_emotion_idx = np.argmax(probabilities)
        
        if method.lower() == 'lime':
            return self.explain_with_lime(text, top_emotion_idx)
        elif method.lower() == 'shap':
            return self.explain_with_shap(text, top_emotion_idx)
        else:
            raise ValueError(f"Unsupported explanation method: {method}. Use 'lime' or 'shap'.")


