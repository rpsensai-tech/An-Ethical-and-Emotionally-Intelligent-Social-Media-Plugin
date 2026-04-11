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
async def health_check(): return HealthResponse(status="healthy", models_loaded={"text": text_service is not None}, services={"text": text_service is not None})

@router.post("/text/enhanced", response_model=EnhancedTextAnalysisResponse)
async def analyze_text_enhanced(request: EnhancedTextAnalysisRequest):
    return EnhancedTextAnalysisResponse(**text_service.predict_enhanced(request.text, request.threshold))

@router.post("/filter/content", response_model=FilterResponse)
async def filter_content(request: FilterContentRequest):
    return FilterResponse(**filtering_service.analyze_content(request.text))

@router.post("/emojis/suggest", response_model=EmojiSuggestionResponse)
async def suggest_emojis(request: EmojiSuggestionRequest):
    return EmojiSuggestionResponse(**emoji_service.suggest_emojis(text=request.text, emotions=request.emotions, limit=request.limit))

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    return ChatResponse(**await chat_service.process_message(request.message, request.context))
