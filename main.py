import logging
# Completely silence huggingface/transformers logging config to wipe out API chatter
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import uvicorn
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

import warnings
from sklearn.exceptions import InconsistentVersionWarning
import transformers
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
transformers.logging.set_verbosity_error()

# Add component paths to sys.path
root_dir = Path(__file__).parent
components_dir = root_dir / "components"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(components_dir / "behavior/behavior_detection_component"))
sys.path.insert(0, str(components_dir / "cyberbullying"))
sys.path.insert(0, str(components_dir / "emotion"))
sys.path.insert(0, str(components_dir / "recommendation"))


# Import routers and startup logic from each component
# Behavior
from components.behavior.behavior_detection_component.app.api import routes as behavior_routes
from components.behavior.behavior_detection_component.app.services.background import background_loop

# Cyberbullying
from components.cyberbullying.app.api import routes as cyberbullying_routes

# Emotion
from components.emotion.app.api import routes as emotion_routes
from components.emotion.app.api.image_api import app as image_app, load_model as load_image_api_model
from components.emotion.assets.configs.config import config as emotion_main_config
from components.emotion.assets.configs.emotion_config import emotion_config
from components.emotion.app.services.text_prediction_service import TextPredictionService
from components.emotion.app.services.filtering_service import EthicalFilteringService
from components.emotion.app.services.emoji_service import emoji_service
from components.emotion.app.services.chat_service import ChatService

# Recommendation
from components.recommendation.app.api import routes as recommendation_routes

# Azure Downloader
from azure_downloader import download_blob_if_not_exists

logging.basicConfig(
    level=getattr(logging, emotion_main_config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def download_emotion_assets():
    """Downloads all models and assets for the emotion component."""
    print("[INFO] Checking for emotion models and assets...")
    
    # Define local paths
    slang_dict_path = components_dir / "emotion/app/data/slang_dictionary.json"
    
    # 1. Download slang dictionary (if it existed in Azure, but it's local only right now, so this will naturally skip/fail gracefully)
    download_blob_if_not_exists("emotion/data/slang_dictionary.json", slang_dict_path)
    
    # 2. Download AffectNet weights
    affectnet_dir = components_dir / "emotion/models/affectnet"
    download_blob_if_not_exists("emotion/models/affectnet/affectnet_emotion_model_weights.pth", affectnet_dir / "affectnet_emotion_model_weights.pth")
    download_blob_if_not_exists("emotion/models/affectnet/affectnet_emotion_model_checkpoint.pth", affectnet_dir / "affectnet_emotion_model_checkpoint.pth")
    
    # 3. Download Text Emotion 'default' model
    default_text_dir = components_dir / "emotion/models/text/default"
    for filename in ["metadata.json", "model.pt"]:
        download_blob_if_not_exists(f"emotion/models/text/default/{filename}", default_text_dir / filename)
        
    for filename in ["merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]:
        download_blob_if_not_exists(f"emotion/models/text/default/tokenizer/{filename}", default_text_dir / "tokenizer" / filename)
        
    # 4. Download Sarcasm Detector
    sarcasm_text_dir = components_dir / "emotion/models/text/sarcasm-detector"
    for filename in ["config.json", "merges.txt", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]:
        download_blob_if_not_exists(f"emotion/models/text/sarcasm-detector/{filename}", sarcasm_text_dir / filename)
    
    print("[INFO] Emotion assets check complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download all assets at startup
    download_emotion_assets()

    # Emotion component startup
    logger.info("Starting emotion services...")
    emotion_main_config.ensure_directories()
    text_service = TextPredictionService(models_dir=emotion_main_config.TEXT_MODELS_DIR, emotions=emotion_config.GOEMOTIONS_EMOTIONS, device=emotion_main_config.DEVICE)
    text_service.load_model("default")
    
    filtering_service = EthicalFilteringService(toxicity_threshold=emotion_main_config.TOXICITY_THRESHOLD)
    
    chat_service_instance = ChatService(text_service=text_service, filtering_service=filtering_service)
    
    emotion_routes.set_services(text_service, filtering_service, emoji_service, chat_service_instance)
    
    load_image_api_model()
    logger.info("Emotion services started.")

    # Behavior component startup
    logger.info("Starting behavior detection background worker...")
    behavior_thread = threading.Thread(target=background_loop, daemon=True)
    behavior_thread.start()
    logger.info("Behavior detection background worker started.")
    
    logger.info("All services started!")
    yield
    logger.info("Application shutdown.")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Unified AI Social Media Plugin",
        description="A centralized AI-powered plugin for social media platforms.",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers from each component with a prefix
    app.include_router(behavior_routes.router, prefix="/behavior", tags=["Behavior Detection"])
    app.include_router(cyberbullying_routes.router, prefix="/cyberbullying", tags=["Cyberbullying Detection"])
    app.include_router(emotion_routes.router, prefix="/emotion", tags=["Emotion Intelligence"])
    app.include_router(recommendation_routes.router, prefix="/recommendation", tags=["Recommendation"])
    
    # Mount the image API from the emotion component
    app.mount("/emotion/image-api", image_app)

    return app

fastapi_app = create_app()

# Also alias it back to 'app' so that things like 'uvicorn main:app' still work
app = fastapi_app

if __name__ == "__main__":
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000, reload=False)
