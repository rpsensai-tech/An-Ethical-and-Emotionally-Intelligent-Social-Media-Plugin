from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
from app.api.routes import router
from app.services.background import background_loop

app = FastAPI(title="Behavior Detection API")

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Include routes
# -------------------------------
app.include_router(router)

# -------------------------------
# Background worker
# -------------------------------
@app.on_event("startup")
def start_background_worker():
    thread = threading.Thread(target=background_loop, daemon=True)
    thread.start()