"""
Unified Emotion Configuration
Maps emotions across modalities and defines reaction rules
"""

from typing import Dict, List, Set
from dataclasses import dataclass, field


@dataclass
class EmotionConfig:
    """Centralized emotion configuration for the entire system"""
    
    # Core 8 emotions used across all modalities
    CORE_EMOTIONS: List[str] = field(default_factory=lambda: [
        "happy", "sad", "angry", "fear", 
        "surprise", "disgust", "neutral", "other"
    ])
    
    # GoEmotions 28 fine-grained emotions (text-based)
    GOEMOTIONS_EMOTIONS: List[str] = field(default_factory=lambda: [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ])
    
    # Map GoEmotions to Core Emotions
    GOEMOTIONS_TO_CORE: Dict[str, str] = field(default_factory=lambda: {
        "joy": "happy",
        "amusement": "happy",
        "excitement": "happy",
        "gratitude": "happy",
        "love": "happy",
        "optimism": "happy",
        "relief": "happy",
        "pride": "happy",
        "admiration": "happy",
        "approval": "happy",
        "caring": "happy",
        "desire": "happy",
        
        "sadness": "sad",
        "grief": "sad",
        "disappointment": "sad",
        "embarrassment": "sad",
        "remorse": "sad",
        
        "anger": "angry",
        "annoyance": "angry",
        "disapproval": "angry",
        
        "fear": "fear",
        "nervousness": "fear",
        
        "surprise": "surprise",
        "realization": "surprise",
        "confusion": "surprise",
        "curiosity": "surprise",
        
        "disgust": "disgust",
        
        "neutral": "neutral",
    })
    
    # Ekman's 6 Basic Emotions
    EKMAN_EMOTIONS: List[str] = field(default_factory=lambda: [
        "joy", "sadness", "anger", "fear", "surprise", "disgust"
    ])
    
    # Map GoEmotions to Ekman
    GOEMOTIONS_TO_EKMAN: Dict[str, str] = field(default_factory=lambda: {
        "joy": "joy",
        "amusement": "joy",
        "excitement": "joy",
        "gratitude": "joy",
        "love": "joy",
        "optimism": "joy",
        "relief": "joy",
        "pride": "joy",
        "admiration": "joy",
        "approval": "joy",
        "caring": "joy",
        "desire": "joy",
        
        "sadness": "sadness",
        "grief": "sadness",
        "disappointment": "sadness",
        "embarrassment": "sadness",
        "remorse": "sadness",
        
        "anger": "anger",
        "annoyance": "anger",
        "disapproval": "anger",
        
        "fear": "fear",
        "nervousness": "fear",
        
        "surprise": "surprise",
        "realization": "surprise",
        "confusion": "surprise",
        "curiosity": "surprise",
        
        "disgust": "disgust",
    })
    
    # Sentiment categories
    SENTIMENT_MAPPING: Dict[str, List[str]] = field(default_factory=lambda: {
        "positive": [
            "amusement", "excitement", "joy", "love", "desire", "optimism", 
            "caring", "pride", "admiration", "gratitude", "relief", "approval"
        ],
        "negative": [
            "fear", "nervousness", "remorse", "embarrassment", "disappointment", 
            "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"
        ],
        "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
        "neutral": ["neutral"]
    })
    
    # Comprehensive emoji mappings by core emotion
    EMOTION_TO_EMOJIS: Dict[str, List[str]] = field(default_factory=lambda: {
        "happy": ["😀", "😄", "😊", "😎", "😂", "🥰", "😍", "❤️", "👍", "🎉", "✨", "🔥", "👏", "💕"],
        "sad": ["😢", "😭", "💔", "😔", "😞", "☹️", "🤍", "💙", "🫂", "🙏"],
        "angry": ["😡", "😤", "😠", "🤬", "😾", "💢", "👎"],
        "fear": ["😨", "😰", "😱", "😧", "😦", "😟", "💙", "🫂", "🙏"],
        "surprise": ["😮", "😲", "😳", "🤯", "🙀", "😯", "✨"],
        "disgust": ["🤢", "🤮", "😒", "🙄", "😖", "🤭"],
        "neutral": ["👍", "🙂", "😐", "😶", "🤝", "👌", "🤔"],
        "other": ["🤔", "😕", "😬", "🤷", "💭"]
    })
    
    # Blocked emojis by emotion (inappropriate reactions)
    EMOTION_BLOCKED_EMOJIS: Dict[str, List[str]] = field(default_factory=lambda: {
        "happy": ["😢", "😭", "😡", "😤", "😰", "😱", "🤢"],
        "sad": ["😂", "😆", "😎", "🎉", "🤣", "😁", "🔥"],
        "angry": ["😂", "😆", "😍", "🥰", "😘", "🎉", "🔥"],
        "fear": ["😂", "😆", "😎", "🎉", "🤣", "🔥"],
        "disgust": ["😍", "😘", "🥰", "❤️", "😂", "💕"],
        "surprise": [],  # Context-dependent
        "neutral": ["😭", "🤯", "😡", "😱"],
        "other": []
    })
    
    @classmethod
    def get_allowed_emojis(cls, emotions: List[str]) -> List[str]:
        """Get union of allowed emojis for detected emotions"""
        allowed = set()
        config = cls()
        for emotion in emotions:
            if emotion in config.EMOTION_TO_EMOJIS:
                allowed.update(config.EMOTION_TO_EMOJIS[emotion])
        return sorted(list(allowed))
    
    @classmethod
    def get_blocked_emojis(cls, emotions: List[str]) -> List[str]:
        """Get union of blocked emojis for detected emotions"""
        blocked = set()
        config = cls()
        for emotion in emotions:
            if emotion in config.EMOTION_BLOCKED_EMOJIS:
                blocked.update(config.EMOTION_BLOCKED_EMOJIS[emotion])
        return sorted(list(blocked))
    
    @classmethod
    def filter_reactions(cls, emotions: List[str], available_emojis: List[str]) -> Dict[str, List[str]]:
        """Filter emoji reactions based on detected emotions"""
        allowed = set(cls.get_allowed_emojis(emotions))
        blocked = set(cls.get_blocked_emojis(emotions))
        
        # Remove blocked emojis from allowed set
        allowed = allowed - blocked
        
        # Filter available emojis
        filtered = [emoji for emoji in available_emojis if emoji in allowed]
        removed = [emoji for emoji in available_emojis if emoji in blocked]
        
        return {
            "allowed": sorted(filtered),
            "blocked": sorted(removed),
            "suggested": sorted(list(allowed))
        }
    
    @classmethod
    def map_to_core_emotion(cls, goemotions: Dict[str, float]) -> Dict[str, float]:
        """Map GoEmotions probabilities to core emotions"""
        core_scores = {}
        config = cls()
        
        for emotion, score in goemotions.items():
            core_emotion = config.GOEMOTIONS_TO_CORE.get(emotion, "other")
            if core_emotion not in core_scores:
                core_scores[core_emotion] = 0.0
            core_scores[core_emotion] = max(core_scores[core_emotion], score)
        
        return core_scores
    
    @classmethod
    def get_sentiment(cls, emotions: Dict[str, float], threshold: float = 0.3) -> str:
        """Determine overall sentiment from emotion probabilities"""
        config = cls()
        sentiment_scores = {"positive": 0.0, "negative": 0.0, "ambiguous": 0.0, "neutral": 0.0}
        
        for emotion, score in emotions.items():
            if score < threshold:
                continue
            for sentiment, emotion_list in config.SENTIMENT_MAPPING.items():
                if emotion in emotion_list:
                    sentiment_scores[sentiment] += score
        
        return max(sentiment_scores, key=sentiment_scores.get) if max(sentiment_scores.values()) > 0 else "neutral"


# Global instance
emotion_config = EmotionConfig()
