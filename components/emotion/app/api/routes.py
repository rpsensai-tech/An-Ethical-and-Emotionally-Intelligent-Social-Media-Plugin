from fastapi import APIRouter, HTTPException
from ..schemas.requests import (
    FilterContentRequest, FilterResponse, HealthResponse,
    EmojiSuggestionRequest, EmojiSuggestionResponse,
    EnhancedTextAnalysisRequest, EnhancedTextAnalysisResponse,
    ChatRequest, ChatResponse
)
import logging

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)

text_service, filtering_service, emoji_service, chat_service = None, None, None, None

def set_services(text_svc, filtering_svc, emoji_svc=None, chat_svc=None):
    global text_service, filtering_service, emoji_service, chat_service
    text_service, filtering_service, emoji_service, chat_service = text_svc, filtering_svc, emoji_svc, chat_svc

@router.get("/health", response_model=HealthResponse)
async def health_check():
    chat_available = bool(chat_service)
    if chat_service and hasattr(chat_service, "get_status"):
        try:
            chat_available = bool(chat_service.get_status().get("available", False))
        except Exception:
            chat_available = False

    services = {
        "text": "ready" if text_service else "unavailable",
        "filtering": "ready" if filtering_service else "unavailable",
        "emoji": "ready" if emoji_service else "unavailable",
        "chat": "ready" if chat_available else "unavailable",
    }
    device = str(getattr(text_service, "device", "unknown")) if text_service else "unknown"
    return HealthResponse(status="healthy", version="1.0.0", services=services, device=device)

@router.post("/text/enhanced", response_model=EnhancedTextAnalysisResponse)
async def analyze_text_enhanced(request: EnhancedTextAnalysisRequest):
    try:
        if not text_service:
            raise Exception("No text service initialized")
        result = text_service.predict_emotions(request.text, request.threshold)
        return EnhancedTextAnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Text prediction failed or models missing: {e}")
        # Return fallback response so the frontend doesn't crash on cross-origin requests
        return EnhancedTextAnalysisResponse(
            text=request.text,
            emotions=[{"emotion": "neutral", "probability": 1.0, "confidence_level": "high"}],
            top_emotion="neutral",
            sentiment="neutral",
            processing_time=0.0
        )

@router.post("/filter/content", response_model=FilterResponse)
async def filter_content(request: FilterContentRequest):
    try:
        if not filtering_service:
            raise Exception("No filtering service initialized")
        return FilterResponse(**filtering_service.analyze_content(request.text))
    except Exception as e:
        logger.error(f"Filtering content failed: {e}")
        return FilterResponse(
            is_harmful=False,
            should_block=False,
            toxicity_score=0.0,
            severity="low",
            categories_detected=[],
            explanation="Fallback: Filter service unavailable.",
            recommendation="none"
        )

@router.post("/emojis/suggest", response_model=EmojiSuggestionResponse)
async def suggest_emojis(request: EmojiSuggestionRequest):
    try:
        if not emoji_service or not text_service:
            raise Exception("No emoji or text service initialized")
            
        # Get emotion mapping first
        text_result = text_service.predict_emotions(request.text, request.threshold)
        emotion_probs = {e['emotion']: e['probability'] for e in text_result['emotions']}
        
        suggestion_data = emoji_service.suggest_emojis(
            text=request.text,
            emotion_probabilities=emotion_probs,
            core_emotions=text_result['core_emotions'],
            sentiment=text_result['sentiment'],
            threshold=request.threshold
        )
        
        # Merge suggestion data with the extra required fields for the response
        response_data = {
            # Fields from text prediction
            "text": request.text,
            "top_emotion": text_result['top_emotion'],
            "top_probability": text_result['top_probability'],
            "detected_emotions": emotion_probs,
            "sentiment": text_result['sentiment'],
            
            # Fields from emoji suggestion
            "allowed_emojis": suggestion_data.get("allowed_emojis", []),
            "blocked_emojis": suggestion_data.get("blocked_emojis", []),
            "suggested_emojis": suggestion_data.get("suggested_emojis", []),
            "emoji_categories": suggestion_data.get("emoji_categories", {}),
            "reasoning": suggestion_data.get("reasoning", "")
        }
        
        return EmojiSuggestionResponse(**response_data)
    except Exception as e:
        logger.error(f"Emoji suggestion failed: {e}")
        # Fallback to prevent frontend CORS from crashing from an unhandled 500 error
        return EmojiSuggestionResponse(
            text=request.text,
            top_emotion="neutral",
            top_probability=1.0,
            detected_emotions={"neutral": 1.0},
            allowed_emojis=[],
            blocked_emojis=[],
            suggested_emojis=[],
            emoji_categories={},
            reasoning="Fallback: Emoji service currently unavailable.",
            sentiment="neutral"
        )

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    try:
        if not chat_service:
            return ChatResponse(
                response="Chat service is unavailable right now.",
                status="unavailable"
            )

        conversation_history = [
            {"role": item.role, "content": item.content}
            for item in request.conversation_history
        ]

        result = await chat_service.chat(
            user_message=request.message,
            conversation_history=conversation_history,
            emotion_context=request.emotion_context,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        return ChatResponse(
            response="I am having trouble responding right now. Please try again.",
            status="error"
        )
