from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="AI Moderation Service",
    description="Cyberbullying detection API for text and memes",
    version="1.0.0"
)

# Include all API routes
app.include_router(router)