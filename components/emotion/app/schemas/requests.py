"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# Request Schemas

class TextPredictionRequest(BaseModel):
    """Request for text emotion prediction"""
    model_config = {"protected_namespaces": ()}
    text: str = Field(..., description="Text content to analyze", min_length=1, max_length=5000)
    threshold: float = Field(default=0.3, description="Emotion probability threshold", ge=0.0, le=1.0)
    model_name: Optional[str] = Field(default="default", description="Model name to use")


class TextExplainRequest(BaseModel):
    """Request for text explainability"""
    model_config = {"protected_namespaces": ()}
    text: str = Field(..., description="Text content to explain", min_length=1, max_length=5000)
    model_name: Optional[str] = Field(default="default", description="Model name to use")
    method: Optional[str] = Field(default="attention", description="Explanation method (attention, lime, shap)")


class EmojiSuggestionRequest(BaseModel):
    """Request for emoji suggestions based on text"""
    text: str = Field(..., description="Text content to analyze", min_length=1, max_length=5000)
    threshold: float = Field(default=0.3, description="Emotion threshold", ge=0.0, le=1.0)


class ReactionSuggestRequest(BaseModel):
    """Request for reaction suggestions"""
    emotion_probabilities: Dict[str, float] = Field(..., description="Emotion probabilities")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    context: str = Field(default="post", description="Content context (post, comment, reply)")


class FilterContentRequest(BaseModel):
    """Request for content filtering"""
    text: str = Field(..., description="Content to filter", min_length=1, max_length=5000)


class FilterSearchRequest(BaseModel):
    """Request for search query filtering"""
    query: str = Field(..., description="Search query to filter", min_length=1, max_length=500)


class SarcasmDetectionRequest(BaseModel):
    """Request for sarcasm detection"""
    text: str = Field(..., description="Text to analyze for sarcasm", min_length=1, max_length=5000)
    threshold: float = Field(default=0.5, description="Sarcasm confidence threshold", ge=0.0, le=1.0)


class SlangDetectionRequest(BaseModel):
    """Request for slang detection"""
    text: str = Field(..., description="Text to analyze for slang", min_length=1, max_length=5000)


class EnhancedTextAnalysisRequest(BaseModel):
    """Request for comprehensive text analysis"""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=5000)
    include_emotions: bool = Field(default=True, description="Include emotion detection")
    include_sarcasm: bool = Field(default=True, description="Include sarcasm detection")
    include_slang: bool = Field(default=True, description="Include slang detection")
    threshold: float = Field(default=0.3, description="Emotion threshold", ge=0.0, le=1.0)


# Response Schemas

class EmotionItem(BaseModel):
    """Individual emotion with probability"""
    emotion: str
    probability: float
    confidence_level: str


class TextPredictionResponse(BaseModel):
    """Response for text emotion prediction"""
    model_config = {"protected_namespaces": ()}
    
    text: str
    emotions: List[EmotionItem]
    significant_emotions: Dict[str, float]
    top_emotion: str
    top_probability: float
    core_emotions: Optional[Dict[str, float]] = None
    sentiment: str
    processing_time: float
    model_used: str
    threshold: float
    modality: str


class ReactionSuggestion(BaseModel):
    """Single reaction suggestion"""
    emoji: str
    reason: str
    priority: int


class ReactionResponse(BaseModel):
    """Response for reaction suggestions"""
    allowed_reactions: List[str]
    blocked_reactions: List[str]
    suggested_reactions: List[ReactionSuggestion]
    top_emotion: str
    top_probability: float
    emotion_category: str
    sentiment: str
    significant_emotions: Dict[str, float]
    reasoning: str


class FilterResponse(BaseModel):
    """Response for content filtering"""
    is_harmful: bool
    should_block: bool
    toxicity_score: float
    severity: str
    categories_detected: List[str]
    explanation: str
    recommendation: str


class SearchFilterResponse(BaseModel):
    """Response for search query filtering"""
    query: str
    allowed: bool
    blocked: bool
    reason: str
    severity: str
    toxicity_score: float
    alternative_suggestion: Optional[str] = None


class EmojiSuggestionResponse(BaseModel):
    """Response for emoji suggestions"""
    text: str
    top_emotion: str
    top_probability: float
    detected_emotions: Dict[str, float]
    allowed_emojis: List[str]
    blocked_emojis: List[str]
    suggested_emojis: List[str]
    emoji_categories: Dict[str, List[str]]
    reasoning: str
    sentiment: str


class ModelInfo(BaseModel):
    """Model information"""
    model_config = {"protected_namespaces": ()}
    model_name: str
    base_model: Optional[str] = None
    backbone: Optional[str] = None
    metrics: Dict[str, Any] = {}
    saved_at: str


class ModelsListResponse(BaseModel):
    """List of available models"""
    text_models: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, str]
    device: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class SarcasmDetectionResponse(BaseModel):
    """Response for sarcasm detection"""
    text: str
    is_sarcastic: bool
    sarcasm_detected: bool
    confidence: float
    sarcasm_probability: float
    non_sarcasm_probability: float
    threshold: float


class SlangDetectionResponse(BaseModel):
    """Response for slang detection"""
    text: str
    has_slang: bool
    slang_detected: bool
    slang_terms: List[str]
    definitions: Dict[str, str]
    slang_count: int
    word_count: int
    slang_density: float


class EnhancedTextAnalysisResponse(BaseModel):
    """Response for comprehensive text analysis"""
    text: str
    emotions: Optional[List[EmotionItem]] = None
    top_emotion: Optional[str] = None
    sentiment: Optional[str] = None
    sarcasm: Optional[Dict[str, Any]] = None
    slang: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    emotion_adjustment: Optional[Dict[str, str]] = None
    processing_time: float


class SafeSearchRequest(BaseModel):
    """Request for ML-based safe search classification"""
    query: str = Field(..., description="Search query to classify", min_length=1, max_length=500)


class SafeSearchResponse(BaseModel):
    """Response for ML-based safe search classification"""
    query: str
    status: str
    message: str
    is_harmful: bool
    scores: Dict[str, float]


# ===== Chat / Emotional Assistant Schemas =====

class ChatMessageItem(BaseModel):
    """A single message in conversation history"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content", min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    """Request for emotional assistant chat"""
    message: str = Field(..., description="User's message", min_length=1, max_length=2000)
    conversation_history: List[ChatMessageItem] = Field(default=[], description="Previous conversation messages")
    emotion_context: Optional[str] = Field(default=None, description="Emotion context from recent posts", max_length=500)


class ChatResponse(BaseModel):
    """Response from emotional assistant"""
    response: str
    status: str
    detected_emotion: Optional[str] = None
    model: Optional[str] = None

class ChatStatusResponse(BaseModel):
    """Chat service status response"""
    available: bool
    model: str
    has_emotion_context: bool
    has_safety_filter: bool
