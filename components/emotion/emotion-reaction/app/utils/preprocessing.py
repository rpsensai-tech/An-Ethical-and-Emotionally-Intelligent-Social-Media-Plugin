"""
Text Preprocessing Utilities
Handles text cleaning and tokenization for emotion analysis
"""

import re
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer
import logging


class TextPreprocessor:
    """Text preprocessing utilities for emotion analysis"""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize the tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.logger.info(f"Tokenizer initialized: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        
        # Replace mentions with [USER]
        text = re.sub(r'@\w+', '[USER]', text)
        
        # Replace Reddit-style usernames
        text = re.sub(r'u/\w+', '[USER]', text)
        text = re.sub(r'r/\w+', '[SUBREDDIT]', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize text using the model's tokenizer"""
        cleaned_text = self.clean_text(text)
        
        encoded = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def tokenize_batch(self, texts: List[str], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        encoded = self.tokenizer(
            cleaned_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_token_words(self, token_ids: torch.Tensor) -> List[str]:
        """Get individual word tokens"""
        return self.tokenizer.convert_ids_to_tokens(token_ids)


class EmotionMapper:
    """Maps emotions between different taxonomies"""
    
    def __init__(self, emotions: List[str], ekman_mapping: Dict[str, List[str]], 
                 sentiment_mapping: Dict[str, List[str]]):
        self.emotions = emotions
        self.ekman_mapping = ekman_mapping
        self.sentiment_mapping = sentiment_mapping
        self.logger = logging.getLogger(__name__)
    
    def get_ekman_emotion(self, emotion_probs: Dict[str, float], threshold: float = 0.1) -> str:
        """Map emotions to Ekman's 6 basic emotions"""
        ekman_scores = {}
        
        for emotion, prob in emotion_probs.items():
            if prob < threshold:
                continue
            
            for ekman_emotion, emotion_list in self.ekman_mapping.items():
                if emotion in emotion_list:
                    if ekman_emotion not in ekman_scores:
                        ekman_scores[ekman_emotion] = 0.0
                    ekman_scores[ekman_emotion] += prob
        
        if not ekman_scores:
            return "neutral"
        
        return max(ekman_scores, key=ekman_scores.get)
    
    def get_sentiment(self, emotion_probs: Dict[str, float], threshold: float = 0.1) -> str:
        """Determine overall sentiment"""
        sentiment_scores = {"positive": 0.0, "negative": 0.0, "ambiguous": 0.0, "neutral": 0.0}
        
        for emotion, prob in emotion_probs.items():
            if prob < threshold:
                continue
            
            for sentiment, emotion_list in self.sentiment_mapping.items():
                if emotion in emotion_list:
                    sentiment_scores[sentiment] += prob
        
        max_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        return max_sentiment if sentiment_scores[max_sentiment] > 0 else "neutral"
    
    def get_emotion_intensity(self, emotion_probs: Dict[str, float]) -> str:
        """Determine emotion intensity level"""
        max_prob = max(emotion_probs.values()) if emotion_probs else 0.0
        
        if max_prob >= 0.8:
            return "very_high"
        elif max_prob >= 0.6:
            return "high"
        elif max_prob >= 0.4:
            return "medium"
        elif max_prob >= 0.2:
            return "low"
        else:
            return "very_low"
