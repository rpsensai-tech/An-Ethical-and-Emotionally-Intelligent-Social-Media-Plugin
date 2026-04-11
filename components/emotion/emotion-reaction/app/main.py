from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(app_dir.parent))

from app.api.image_api import app as image_app, load_model as load_image_api_model
from assets.configs.config import config
from app.api.routes import router, set_services
from app.services.text_prediction_service import TextPredictionService
from app.services.filtering_service import EthicalFilteringService
from assets.configs.emotion_config import emotion_config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    text_service = TextPredictionService(models_dir=config.TEXT_MODELS_DIR, emotions=emotion_config.GOEMOTIONS_EMOTIONS, device=config.DEVICE)
    text_service.load_model("default")
    
    filtering_service = EthicalFilteringService(toxicity_threshold=config.TOXICITY_THRESHOLD)
    
    from app.services.emoji_service import emoji_service
    
    from app.services.chat_service import ChatService
    chat_service_instance = ChatService(text_service=text_service, filtering_service=filtering_service)
    
    set_services(text_service, filtering_service, emoji_service, chat_service_instance)
    
    load_image_api_model()
    
    logger.info("All services started!")
    yield

def create_app() -> FastAPI:
    config.ensure_directories()
    app = FastAPI(title=config.API_TITLE, docs_url="/docs", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    app.mount("/image-api", image_app)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=config.DEBUG, log_level=config.LOG_LEVEL.lower())
