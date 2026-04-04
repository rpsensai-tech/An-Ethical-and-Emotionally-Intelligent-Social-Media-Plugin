from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import shutil
import os
import tempfile

from core.inference import predict_meme, predict_comment_text

app = FastAPI()

# =========================================================
# COMMENT REQUEST MODEL
# =========================================================

class CommentRequest(BaseModel):
    text: str


# =========================================================
# IMAGE MODERATION ENDPOINT
# =========================================================

@app.post("/moderate_meme")
async def moderate_meme(
    image: UploadFile = File(...),
    caption: str = Form("")
):

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(image.file, tmp)
        temp_path = tmp.name

    try:
        result = predict_meme(temp_path, caption)
        return result

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================
# COMMENT MODERATION ENDPOINT
# =========================================================

@app.post("/moderate_comment")
async def moderate_comment(data: dict):
    text = (data.get("text") or "").strip()

    print("---- COMMENT API DEBUG ----")
    print("RAW DATA:", data)
    print("TEXT:", text)

    if not text:
        result = {
            "prediction": "non-bullying",
            "probability": 0.0,
            "severity": "NON_BULLYING",
            "recommended_action": "ALLOW",
            "reason": "Empty comment",
            "evidence": {
                "text": "",
                "keywords_detected": [],
                "keyword_count": 0,
            },
        }
        print("COMMENT RESULT:", result)
        print("---------------------------")
        return result

    result = predict_comment_text(text)

    print("COMMENT RESULT:", result)
    print("---------------------------")

    return result


