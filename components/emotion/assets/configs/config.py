"""
Backend Configuration
Centralized settings for the emotion-aware social media platform
"""

import os
from pathlib import Path
from typing import List

class Config:
    """Application configuration settings"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    ROOT_DIR = BASE_DIR.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODELS_DIR = ROOT_DIR / "models"
    SHARED_DIR = ROOT_DIR / "shared"
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # Text Model settings
    TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "bert-base-uncased")
    TEXT_MAX_LENGTH = int(os.getenv("TEXT_MAX_LENGTH", "128"))
    TEXT_BATCH_SIZE = int(os.getenv("TEXT_BATCH_SIZE", "16"))
    
    # Image Model settings
    
    # Training settings
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
    
    # Device settings
    DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, cuda
    
    # Emotion thresholds
    EMOTION_THRESHOLD = float(os.getenv("EMOTION_THRESHOLD", "0.3"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    
    # Dataset paths
    GOEMOTIONS_DIR = DATA_DIR / "goemotions"
    
    # Model paths
    TEXT_MODELS_DIR = MODELS_DIR / "text"
    
    # CORS settings
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://localhost",
        "http://localhost:80",
        "http://127.0.0.1",
        "http://localhost/ossn",
    ]
    
    # API settings
    API_PREFIX = "/api"
    API_VERSION = "v1"
    API_TITLE = "Emotion-Aware Social Media Platform API"
    API_DESCRIPTION = """
    An ethical and emotionally intelligent social media platform using responsible and explainable AI.
    
    Features:
    - Context-aware emoji reaction filtering
    - Explainable AI with SHAP and LIME
    - Proactive ethical content filtering
    - Multi-modal emotion analysis
    """
    
    # Filtering settings
    ENABLE_ETHICAL_FILTERING = os.getenv("ENABLE_ETHICAL_FILTERING", "true").lower() == "true"
    TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.7"))
    
    # Explainability settings
    ENABLE_SHAP = os.getenv("ENABLE_SHAP", "true").lower() == "true"
    ENABLE_LIME = os.getenv("ENABLE_LIME", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.TEXT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.GOEMOTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get configuration summary for logging"""
        return {
            "server": {
                "host": cls.HOST,
                "port": cls.PORT,
                "debug": cls.DEBUG
            },
            "models": {
                "text_model": cls.TEXT_MODEL_NAME,
                "device": cls.DEVICE
            },
            "paths": {
                "data_dir": str(cls.DATA_DIR),
                "models_dir": str(cls.MODELS_DIR)
            },
            "features": {
                "ethical_filtering": cls.ENABLE_ETHICAL_FILTERING,
                "shap": cls.ENABLE_SHAP,
                "lime": cls.ENABLE_LIME
            }
        }


config = Config()
