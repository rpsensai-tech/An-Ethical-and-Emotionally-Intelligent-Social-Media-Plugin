import pickle
import re
from pathlib import Path
from typing import Dict, List

import easyocr
import numpy as np
import torch
from PIL import Image
from rapidfuzz import fuzz
from transformers import (
    CLIPModel,
    CLIPProcessor,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from azure_downloader import download_blob_if_not_exists

# =========================================================
# PATHS (UPDATED FOR PROJECT STRUCTURE)
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

CLIP_CLASSIFIER_PATH = MODELS_DIR / "cyberbullying_clip_logreg.pkl"
TEXT_MODEL_PATH = MODELS_DIR / "distilbert_text_only_clean.pt"
COMMENT_MODEL_PATH = MODELS_DIR / "best_distilbert.pt"
TOXIC_WORDS_PATH = ASSETS_DIR / "toxic_words.txt"
SEVERE_WORDS_PATH = ASSETS_DIR / "severe_words.txt"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# AZURE BLOB DOWNLOAD
# =========================================================

def download_cyberbullying_assets():
    """Downloads all models and assets for the cyberbullying component."""
    print("[INFO] Checking for cyberbullying models and assets...")
    # Models
    download_blob_if_not_exists("cyberbullying/models/cyberbullying_clip_logreg.pkl", CLIP_CLASSIFIER_PATH)
    download_blob_if_not_exists("cyberbullying/models/distilbert_text_only_clean.pt", TEXT_MODEL_PATH)
    download_blob_if_not_exists("cyberbullying/models/best_distilbert.pt", COMMENT_MODEL_PATH)
    # Assets
    download_blob_if_not_exists("cyberbullying/assets/toxic_words.txt", TOXIC_WORDS_PATH)
    download_blob_if_not_exists("cyberbullying/assets/severe_words.txt", SEVERE_WORDS_PATH)
    print("[INFO] Cyberbullying assets check complete.")

# Trigger the download on module load
download_cyberbullying_assets()

# =========================================================
# LOAD TOXIC WORDS
# =========================================================

def load_toxic_words(path: Path) -> List[str]:

    words = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:

            w = line.strip().lower()

            if not w:
                continue

            if re.match(r"^\d+\.\d+$", w):
                continue

            words.append(w)

    print(f"[INFO] Loaded toxic dictionary with {len(words)} words")

    return words


TOXIC_WORDS = load_toxic_words(TOXIC_WORDS_PATH)

# =========================================================
# LOAD SEVERE WORDS (UPDATED)
# =========================================================

def load_severe_words(path: Path) -> set:

    words = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:

            w = line.strip().lower()

            if not w:
                continue

            words.add(w)

    print(f"[INFO] Loaded severe words with {len(words)} entries")

    return words


SEVERE_WORDS = load_severe_words(SEVERE_WORDS_PATH)

# =========================================================
# LOAD MODELS
# =========================================================

print(f"[INFO] Using device: {DEVICE}")

# CLIP
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()

with open(CLIP_CLASSIFIER_PATH, "rb") as f:
    clip_classifier = pickle.load(f)

# OCR
ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# =========================================================
# MEME TEXT MODEL
# =========================================================

tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)

text_model = DistilBertForSequenceClassification.from_pretrained(
    DISTILBERT_MODEL_NAME,
    num_labels=2
)

checkpoint = torch.load(TEXT_MODEL_PATH, map_location=DEVICE)
text_model.load_state_dict(checkpoint["model_state"])

text_model.to(DEVICE)
text_model.eval()

print("[INFO] Meme DistilBERT model loaded")

# =========================================================
# COMMENT MODEL
# =========================================================

comment_tokenizer = DistilBertTokenizerFast.from_pretrained(
    DISTILBERT_MODEL_NAME
)

comment_model = DistilBertForSequenceClassification.from_pretrained(
    DISTILBERT_MODEL_NAME,
    num_labels=2
)

comment_state = torch.load(COMMENT_MODEL_PATH, map_location=DEVICE)

comment_model.load_state_dict(comment_state)

comment_model.to(DEVICE)
comment_model.eval()

print("[INFO] Comment DistilBERT model loaded")

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text: str) -> str:

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = text.replace("0", "o").replace("1", "i").replace("3", "e")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(text.split())

    return text

# =========================================================
# OCR
# =========================================================

def extract_ocr_text(image_path: str) -> str:

    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 1024))

        results = ocr_reader.readtext(
            np.array(image),
            detail=0,
            paragraph=True
        )

        if not results:
            return ""

        text = " ".join(results)

        return clean_text(text)

    except Exception as e:
        print(f"[WARN] OCR failed: {e}")
        return ""

# =========================================================
# CLIP + LOGREG
# =========================================================

def get_clip_embedding(image_path: str, text: str) -> np.ndarray:

    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():

        image_features = clip_model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        text_features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    image_features = image_features.pooler_output
    image_features = image_features.cpu().detach().numpy()

    text_features = text_features.pooler_output
    text_features = text_features.cpu().detach().numpy()

    embedding = np.concatenate([image_features, text_features], axis=1)

    return embedding.astype(np.float32)


def predict_clip_score(image_path: str, combined_text: str) -> float:

    embedding = get_clip_embedding(image_path, combined_text)

    proba = clip_classifier.predict_proba(embedding)[0]

    classes = list(clip_classifier.classes_)

    if 1 in classes:
        return float(proba[classes.index(1)])

    return float(proba[1])

# =========================================================
# DISTILBERT (MEME)
# =========================================================

def predict_text_score(text: str) -> float:

    if not text.strip():
        return 0.0

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return float(probs[0][1])

# =========================================================
# DISTILBERT (COMMENT)
# =========================================================

def predict_comment_model_score(text: str) -> float:

    if not text.strip():
        return 0.0

    inputs = comment_tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = comment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return float(probs[0][1])

# =========================================================
# KEYWORDS
# =========================================================

def detect_keywords_fuzzy(text: str, threshold: int = 80) -> List[str]:

    text = clean_text(text)
    words = text.split()
    hits = set()

    for toxic_word in TOXIC_WORDS:

        toxic_word = toxic_word.strip().lower()

        if not toxic_word:
            continue

        if toxic_word in text:
            hits.add(toxic_word)
            continue

        for w in words:
            if fuzz.ratio(toxic_word, w) >= threshold:
                hits.add(toxic_word)
                break

    return list(hits)


def keyword_score_from_hits(hits: List[str]) -> float:

    score = 0

    for word in hits:
        if word in SEVERE_WORDS:
            score += 0.8
        else:
            score += 0.25

    return min(score, 1.0)

# =========================================================
# FUSION
# =========================================================

def compute_final_score(clip_score, text_score, kw_score):

    final_score = (
        0.35 * clip_score +
        0.45 * text_score +
        0.20 * kw_score
    )

    return float(np.clip(final_score, 0, 1))

# =========================================================
# POLICY
# =========================================================

def moderation_policy(score):

    if score < 0.45:
        return "non-bullying", "NON_BULLYING", "ALLOW"
    elif score < 0.75:
        return "bullying", "MILD_BULLYING", "WARN"
    else:
        return "bullying", "BULLYING", "DELETE"

# =========================================================
# EXPLAINABILITY
# =========================================================

def build_reason(prediction, clip_score, text_score, keywords):

    if keywords:
        return f"Toxic language detected: {', '.join(keywords[:5])}"

    if prediction == "bullying":

        if text_score > 0.8:
            return "Text classifier detected strong harmful language."

        if clip_score > 0.8:
            return "Image-text model detected bullying cues."

        return "Multimodal signals indicate harmful content."

    return "No bullying evidence detected"

# =========================================================
# MEME MODERATION
# =========================================================

def predict_meme(image_path: str, caption: str = "") -> Dict:

    image_path = str(image_path)
    caption = clean_text(caption) if caption else ""

    ocr_text = extract_ocr_text(image_path)
    combined_text = f"{caption} {ocr_text}".strip()

    clip_score = predict_clip_score(image_path, combined_text)
    text_score = predict_text_score(combined_text)

    keywords = detect_keywords_fuzzy(combined_text)
    kw_score = keyword_score_from_hits(keywords)

    for word in keywords:
        if word in SEVERE_WORDS:

            return {
                "prediction": "bullying",
                "probability": 0.95,
                "clip_score": clip_score,
                "text_score": text_score,
                "keyword_score": kw_score,
                "severity": "BULLYING",
                "recommended_action": "DELETE",
                "reason": f"Severe abusive word detected: {word}",
                "evidence": {
                    "caption": caption,
                    "ocr_text": ocr_text,
                    "combined_text": combined_text,
                    "keywords_detected": keywords,
                    "keyword_count": len(keywords),
                },
            }

    final_score = compute_final_score(clip_score, text_score, kw_score)

    prediction, severity, action = moderation_policy(final_score)
    reason = build_reason(prediction, clip_score, text_score, keywords)

    return {
        "prediction": prediction,
        "probability": final_score,
        "clip_score": clip_score,
        "text_score": text_score,
        "keyword_score": kw_score,
        "severity": severity,
        "recommended_action": action,
        "reason": reason,
        "evidence": {
            "caption": caption,
            "ocr_text": ocr_text,
            "combined_text": combined_text,
            "keywords_detected": keywords,
            "keyword_count": len(keywords),
        },
    }

# =========================================================
# COMMENT MODERATION
# =========================================================

def predict_comment_text(text: str) -> Dict:

    text = clean_text(text)

    text_score = predict_comment_model_score(text)
    keywords = detect_keywords_fuzzy(text)
    kw_score = keyword_score_from_hits(keywords)

    for word in keywords:
        if word in SEVERE_WORDS:

            return {
                "prediction": "bullying",
                "probability": 0.95,
                "text_score": text_score,
                "keyword_score": kw_score,
                "severity": "BULLYING",
                "recommended_action": "DELETE",
                "reason": f"Severe abusive term detected: {word}",
                "evidence": {
                    "text": text,
                    "keywords_detected": keywords,
                    "keyword_count": len(keywords),
                },
            }

    final_score = 0.80 * text_score + 0.20 * kw_score
    final_score = float(np.clip(final_score, 0, 1))

    if final_score < 0.40:
        prediction = "non-bullying"
        severity = "NON_BULLYING"
        action = "ALLOW"
    elif final_score < 0.65:
        prediction = "bullying"
        severity = "MILD_BULLYING"
        action = "WARN"
    else:
        prediction = "bullying"
        severity = "BULLYING"
        action = "DELETE"

    if keywords:
        reason = f"Toxic language detected: {', '.join(keywords[:5])}"
    elif text_score > 0.85:
        reason = "Fine-tuned DistilBERT detected strong harmful language."
    elif prediction == "bullying":
        reason = "Text moderation signals indicate harmful content."
    else:
        reason = "No bullying evidence detected"

    return {
        "prediction": prediction,
        "probability": final_score,
        "text_score": text_score,
        "keyword_score": kw_score,
        "severity": severity,
        "recommended_action": action,
        "reason": reason,
        "evidence": {
            "text": text,
            "keywords_detected": keywords,
            "keyword_count": len(keywords),
        },
    }