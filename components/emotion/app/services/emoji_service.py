"""
Emoji Suggestion Service
Provides context-aware emoji recommendations based on emotion analysis
"""

from typing import Dict, List, Any, Set
import logging
import sys
from pathlib import Path

from assets.configs.emotion_config import EmotionConfig


class EmojiService:
    """Provides intelligent emoji suggestions based on emotional context"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = EmotionConfig()
        
        # Extended emoji mappings with categories
        self.emoji_categories = {
            "celebration": ["🎉", "🎊", "🥳", "🎈", "🎁", "🍾", "🥂", "✨", "🌟", "💫"],
            "love_affection": ["❤️", "💕", "💖", "💗", "💓", "💞", "💘", "💝", "😍", "🥰", "😘", "💋"],
            "happiness": ["😀", "😄", "😊", "😁", "🙂", "😎", "😆", "😂", "🤣", "☺️"],
            "support": ["🫂", "🤗", "💪", "👍", "👏", "🙌", "🤝", "💙", "🙏"],
            "sadness": ["😢", "😭", "💔", "😔", "😞", "☹️", "😿", "🥺"],
            "anger": ["😡", "😠", "😤", "🤬", "😾", "💢", "👎", "😒"],
            "fear_anxiety": ["😨", "😰", "😱", "😧", "😦", "😟", "😬"],
            "surprise": ["😮", "😲", "😳", "🤯", "🙀", "😯", "😦"],
            "disgust": ["🤢", "🤮", "😒", "🙄", "😖", "😫"],
            "thinking": ["🤔", "💭", "🧐", "🤨", "💡"],
            "neutral": ["😐", "😑", "😶", "🙂", "👌", "🆗"],
            "fire_hot": ["🔥", "💥", "⚡", "🌶️"],
        }
        
    def suggest_emojis(self, text: str, emotion_probabilities: Dict[str, float], 
                      core_emotions: Dict[str, float], sentiment: str,
                      threshold: float = 0.3) -> Dict[str, Any]:
        """
        Generate comprehensive emoji suggestions
        
        Args:
            text: The input text
            emotion_probabilities: GoEmotions probabilities
            core_emotions: Core emotion mappings
            sentiment: Overall sentiment (positive/negative/neutral)
            threshold: Emotion significance threshold
        
        Returns:
            Dictionary with allowed, blocked, suggested emojis and reasoning
        """
        
        # Get significant emotions
        significant = {e: p for e, p in emotion_probabilities.items() if p >= threshold}
        if not significant:
            top_emotion = max(emotion_probabilities, key=emotion_probabilities.get)
            significant = {top_emotion: emotion_probabilities[top_emotion]}
        
        # Map to core emotions
        core_emotion_list = [
            self.config.GOEMOTIONS_TO_CORE.get(e, "other") 
            for e in significant.keys()
        ]
        core_emotion_list = list(set(core_emotion_list))
        
        # Get base allowed and blocked emojis
        allowed = set()
        blocked = set()
        
        for core_emotion in core_emotion_list:
            if core_emotion in self.config.EMOTION_TO_EMOJIS:
                allowed.update(self.config.EMOTION_TO_EMOJIS[core_emotion])
            if core_emotion in self.config.EMOTION_BLOCKED_EMOJIS:
                blocked.update(self.config.EMOTION_BLOCKED_EMOJIS[core_emotion])
        
        # Remove blocked from allowed
        allowed = allowed - blocked
        
        # Add context-aware suggestions based on sentiment
        if sentiment == "positive":
            allowed.update(self.emoji_categories["celebration"])
            allowed.update(self.emoji_categories["support"])
            blocked.update(self.emoji_categories["sadness"])
            blocked.update(self.emoji_categories["anger"])
        elif sentiment == "negative":
            allowed.update(self.emoji_categories["support"])
            blocked.update(self.emoji_categories["happiness"])
            blocked.update(self.emoji_categories["celebration"])
            blocked.update(self.emoji_categories["fire_hot"])
        
        # Remove blocked from allowed (again after additions)
        allowed = allowed - blocked
        
        # Categorize suggested emojis
        categorized_suggestions = {}
        for category, emojis in self.emoji_categories.items():
            category_emojis = [e for e in emojis if e in allowed]
            if category_emojis:
                categorized_suggestions[category] = category_emojis
        
        # Generate top suggestions (most relevant)
        top_suggestions = self._rank_emojis(allowed, significant, core_emotions, sentiment)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(significant, sentiment, core_emotion_list)
        
        return {
            "allowed_emojis": sorted(list(allowed))[:50],  # Limit for UI
            "blocked_emojis": sorted(list(blocked))[:30],
            "suggested_emojis": top_suggestions[:15],
            "emoji_categories": categorized_suggestions,
            "reasoning": reasoning,
            "total_allowed": len(allowed),
            "total_blocked": len(blocked)
        }
    
    def _rank_emojis(self, emojis: Set[str], emotions: Dict[str, float], 
                     core_emotions: Dict[str, float], sentiment: str) -> List[str]:
        """Rank emojis by relevance to detected emotions"""
        
        # Score each emoji based on emotional relevance
        emoji_scores = {}
        
        for emoji in emojis:
            score = 0.0
            
            # Check which categories the emoji belongs to
            for category, category_emojis in self.emoji_categories.items():
                if emoji in category_emojis:
                    # Boost score based on emotion match
                    if sentiment == "positive" and category in ["happiness", "celebration", "love_affection"]:
                        score += 3.0
                    elif sentiment == "negative" and category in ["sadness", "support"]:
                        score += 3.0
                    elif category == "support":
                        score += 2.0
                    elif category == "thinking":
                        score += 1.0
            
            # Boost based on core emotion strength
            for emotion, prob in core_emotions.items():
                if emotion in self.config.EMOTION_TO_EMOJIS:
                    if emoji in self.config.EMOTION_TO_EMOJIS[emotion]:
                        score += prob * 5.0
            
            emoji_scores[emoji] = score
        
        # Sort by score
        ranked = sorted(emoji_scores.items(), key=lambda x: x[1], reverse=True)
        return [emoji for emoji, score in ranked if score > 0]
    
    def _generate_reasoning(self, emotions: Dict[str, float], sentiment: str, 
                           core_emotions: List[str]) -> str:
        """Generate human-readable explanation for emoji suggestions"""
        
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        emotion_names = [e[0] for e in top_emotions]
        
        reasoning_parts = []
        
        # Main emotion explanation
        if len(emotion_names) == 1:
            reasoning_parts.append(f"Your text expresses {emotion_names[0]}.")
        elif len(emotion_names) == 2:
            reasoning_parts.append(f"Your text expresses {emotion_names[0]} and {emotion_names[1]}.")
        else:
            reasoning_parts.append(f"Your text expresses {', '.join(emotion_names[:-1])}, and {emotion_names[-1]}.")
        
        # Sentiment explanation
        if sentiment == "positive":
            reasoning_parts.append("We've suggested joyful, celebratory, and supportive emojis.")
            reasoning_parts.append("Sad or angry emojis are blocked to maintain positive tone.")
        elif sentiment == "negative":
            reasoning_parts.append("We've suggested empathetic and supportive emojis.")
            reasoning_parts.append("Joyful or celebratory emojis are blocked as they'd be inappropriate.")
        else:
            reasoning_parts.append("We've suggested neutral and thoughtful emojis.")
        
        return " ".join(reasoning_parts)


# Global instance
emoji_service = EmojiService()
