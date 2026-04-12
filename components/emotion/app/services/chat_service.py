"""
Emotional Assistant Chat Service
Uses Groq LLM with PII stripping, safety guardrails, and emotion context.
"""

import re
import os
from dotenv import load_dotenv
load_dotenv()
import logging
from typing import List, Dict, Optional
from groq import Groq

logger = logging.getLogger(__name__)

# PII patterns to strip before sending to LLM
PII_PATTERNS = [
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
    # Phone numbers (international and local formats)
    (r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', '[PHONE]'),
    # Credit card numbers
    (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CARD_NUMBER]'),
    # SSN / National IDs
    (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', '[ID_NUMBER]'),
    # IP addresses
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]'),
    # URLs with potential PII
    (r'https?://[^\s]+', '[URL]'),
    # Physical addresses (simple pattern: number + street)
    (r'\b\d{1,5}\s+(?:[A-Za-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Circle|Cir)\b', '[ADDRESS]'),
    # Date of birth patterns
    (r'\b(?:DOB|date of birth|born on|birthday)[:\s]+[\d/\-\.]+\b', '[DOB]'),
    # Passport numbers
    (r'\b[A-Z]{1,2}\d{6,9}\b', '[PASSPORT]'),
]

# Harmful content patterns to block
HARMFUL_PATTERNS = [
    r'\b(?:kill|murder|suicide|self[-\s]?harm)\b',
    r'\b(?:bomb|weapon|explosive|attack)\b',
    r'\b(?:hack|exploit|breach|ddos)\b',
    r'\b(?:drug|narcotic|cocaine|heroin|meth)\b.*(?:buy|sell|make|cook|deal)',
    r'\b(?:hate|racist|sexist)\b.*\b(?:should|deserve|need)\b',
]

SYSTEM_PROMPT = """You are MoodBuddy, a friendly and empathetic emotional wellness assistant on a social network. Your role is to:

1. Help users understand and process their emotions in a healthy way
2. Provide supportive, caring responses that validate feelings
3. Suggest healthy coping strategies when appropriate
4. Encourage positive social interactions
5. Be warm, conversational, and approachable

STRICT RULES:
- NEVER provide medical, legal, or financial advice. Redirect to professionals.
- NEVER engage with harmful, violent, hateful, or sexual content. Politely decline.
- NEVER generate or discuss personal information about anyone.
- NEVER help with anything illegal or unethical.
- If someone seems in crisis, gently suggest professional resources like crisis helplines.
- Keep responses concise (2-4 sentences typically) unless the user needs more detail.
- Stay positive and constructive. Focus on emotional well-being.
- You can discuss emotions, feelings, mood, relationships (in general terms), self-care, and mindfulness.
- If you detect the user's emotional state from context, acknowledge it empathetically.

You have access to emotion analysis data about the user's recent posts. Use this context to be more empathetic and relevant, but NEVER reveal raw analysis data or scores to the user.

Remember: You are a supportive companion, NOT a therapist or counselor."""


def strip_pii(text: str) -> str:
    """Remove personally identifiable information from text before sending to LLM."""
    cleaned = text
    for pattern, replacement in PII_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def is_harmful_request(text: str) -> bool:
    """Check if the user's message contains harmful intent."""
    text_lower = text.lower()
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def is_safe_response(response: str) -> bool:
    """Validate that the LLM response doesn't contain harmful content."""
    harmful_indicators = [
        r'\b(?:kill yourself|end your life|hurt yourself)\b',
        r'\b(?:how to make|instructions for).*(?:bomb|weapon|drug|poison)\b',
        r'\b(?:password|credit card|social security)\b.*\b(?:is|number)\b',
    ]
    for pattern in harmful_indicators:
        if re.search(pattern, response, re.IGNORECASE):
            return False
    return True


class ChatService:
    """Emotional assistant chat service using Groq LLM."""

    def __init__(self, api_key: str = None, text_service=None, filtering_service=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("No GROQ_API_KEY found. Chat service will be unavailable.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

        self.text_service = text_service
        self.filtering_service = filtering_service
        self.model = "llama-3.3-70b-versatile"
        self.max_history = 10  # Keep last N messages for context

    def _build_emotion_context(self, user_message: str) -> str:
        """Analyze the user's message for emotions and build context for LLM."""
        if not self.text_service:
            return ""

        try:
            prediction = self.text_service.predict(user_message, threshold=0.2)
            if not prediction or "top_emotions" not in prediction:
                return ""

            top_emotions = prediction.get("top_emotions", [])
            if not top_emotions:
                return ""

            emotion_strs = []
            for e in top_emotions[:3]:
                emotion_strs.append(f"{e['emotion']} ({e['probability']:.0%})")

            sentiment = prediction.get("sentiment", {})
            sentiment_label = sentiment.get("label", "neutral") if isinstance(sentiment, dict) else "neutral"

            return (
                f"\n[Emotion Context: The user's message conveys {', '.join(emotion_strs)}. "
                f"Overall sentiment: {sentiment_label}. "
                f"Use this to be more empathetic but don't mention these scores.]"
            )
        except Exception as e:
            logger.debug(f"Emotion analysis for chat context failed: {e}")
            return ""

    def _check_content_safety(self, text: str) -> Dict:
        """Check text for harmful content using the filtering service."""
        if not self.filtering_service:
            return {"is_harmful": False}

        try:
            result = self.filtering_service.filter_content(text)
            return result
        except Exception as e:
            logger.debug(f"Content safety check failed: {e}")
            return {"is_harmful": False}

    async def chat(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        emotion_context: str = None,
    ) -> Dict:
        """
        Process a chat message and return LLM response.

        Args:
            user_message: The user's message
            conversation_history: Previous messages [{"role": "user"/"assistant", "content": "..."}]
            emotion_context: Optional emotion context from recent posts

        Returns:
            Dict with response, detected_emotion, etc.
        """
        if not self.client:
            return {
                "response": "I'm sorry, the chat service is currently unavailable. Please try again later! 💙",
                "status": "unavailable",
            }

        # Step 1: Check for harmful input
        if is_harmful_request(user_message):
            return {
                "response": (
                    "I appreciate you reaching out, but I'm not able to help with that kind of request. "
                    "If you're going through a tough time, please consider reaching out to a professional. "
                    "You can contact crisis helplines like 988 (Suicide & Crisis Lifeline) anytime. 💙"
                ),
                "status": "blocked",
                "reason": "harmful_content",
            }

        # Step 2: Run content safety filter
        safety = self._check_content_safety(user_message)
        if safety.get("is_harmful"):
            return {
                "response": (
                    "I want to keep our conversation positive and supportive. "
                    "Could we talk about something that helps you feel better? "
                    "I'm here to help with emotions, mood, and well-being! 😊"
                ),
                "status": "filtered",
                "reason": "content_filter",
            }

        # Step 3: Strip PII from the user message
        clean_message = strip_pii(user_message)

        # Step 4: Build emotion context from user's message
        msg_emotion_context = self._build_emotion_context(clean_message)

        # Step 5: Build the messages array for Groq
        system_content = SYSTEM_PROMPT
        if emotion_context:
            system_content += f"\n\n[Recent Post Context: {strip_pii(emotion_context)}]"
        if msg_emotion_context:
            system_content += msg_emotion_context

        messages = [{"role": "system", "content": system_content}]

        # Add conversation history (also strip PII from history)
        if conversation_history:
            for msg in conversation_history[-self.max_history :]:
                role = msg.get("role", "user")
                content = strip_pii(msg.get("content", ""))
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # Add current message
        messages.append({"role": "user", "content": clean_message})

        # Step 6: Call Groq API
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                top_p=0.9,
                stop=None,
            )

            response_text = completion.choices[0].message.content.strip()

            # Step 7: Validate response safety
            if not is_safe_response(response_text):
                response_text = (
                    "I want to make sure our chat stays positive and helpful. "
                    "How are you feeling today? I'd love to help you explore your emotions! 😊"
                )

            # Step 8: Extract detected emotion for UI
            detected_emotion = None
            if msg_emotion_context:
                try:
                    prediction = self.text_service.predict(clean_message, threshold=0.2)
                    if prediction and prediction.get("top_emotions"):
                        detected_emotion = prediction["top_emotions"][0]["emotion"]
                except Exception:
                    pass

            return {
                "response": response_text,
                "status": "success",
                "detected_emotion": detected_emotion,
                "model": self.model,
            }

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {
                "response": (
                    "Oops! I had a little hiccup. Could you try sending that again? "
                    "I'm here and ready to chat! 💙"
                ),
                "status": "error",
                "error": str(e),
            }

    def get_status(self) -> Dict:
        """Get chat service status."""
        return {
            "available": self.client is not None,
            "model": self.model,
            "has_emotion_context": self.text_service is not None,
            "has_safety_filter": self.filtering_service is not None,
        }
