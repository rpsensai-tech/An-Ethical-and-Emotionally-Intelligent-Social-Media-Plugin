from fastapi import APIRouter, Query
from pydantic import BaseModel
import json
import os
from datetime import datetime
from assets.ossn_adapter import fetch_features
from core.inference import predict_user, predict_batch

router = APIRouter()

# =========================================================
# PATH SETUP
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNTIME_PATH = os.path.join(BASE_DIR, "assets", "runtime")
RESTRICTIONS_FILE = os.path.join(RUNTIME_PATH, "behaviourguard_restrictions.json")
HISTORY_FILE = os.path.join(RUNTIME_PATH, "behaviourguard_history.json")


# =========================================================
# HELPERS
# =========================================================

def load_restrictions():
    os.makedirs(RUNTIME_PATH, exist_ok=True)

    if not os.path.exists(RESTRICTIONS_FILE):
        with open(RESTRICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}

    try:
        with open(RESTRICTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_restrictions(data):
    os.makedirs(RUNTIME_PATH, exist_ok=True)
    with open(RESTRICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_history(result):
    os.makedirs(RUNTIME_PATH, exist_ok=True)

    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        history = []

    history.append({
        "user_id": result["user_id"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "timestamp": result["timestamp"]
    })

    history = history[-1000:]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# =========================================================
# REQUEST MODELS
# =========================================================

class ModerationRequest(BaseModel):
    user_id: int
    action: str
    mode: str

class AnalyzeUserRequest(BaseModel):
    user_id: int


# =========================================================
# ROUTES
# =========================================================

@router.get("/")
def root():
    return {"message": "Behavior Detection API running"}


@router.post("/moderation")
def moderation(req: ModerationRequest):
    restrictions = load_restrictions()

    user_id = str(req.user_id)
    action = req.action.strip()
    mode = req.mode.strip()

    if action not in ["limit_posting", "limit_comments"]:
        return {"status": "error", "message": "Invalid action"}

    if mode not in ["apply", "disable"]:
        return {"status": "error", "message": "Invalid mode"}

    if user_id not in restrictions or not isinstance(restrictions[user_id], dict):
        restrictions[user_id] = {}

    if mode == "apply":
        restrictions[user_id][action] = True

    if mode == "disable":
        if action in restrictions[user_id]:
            del restrictions[user_id][action]
        if not restrictions[user_id]:
            del restrictions[user_id]

    save_restrictions(restrictions)

    return {
        "status": "success",
        "message": "Moderation updated",
        "restrictions": restrictions
    }


@router.get("/check_restriction")
def check_restriction(user_id: int = Query(...), type: str = Query(...)):
    restrictions = load_restrictions()
    restricted = False

    uid = str(user_id)
    if uid in restrictions and type in restrictions[uid]:
        restricted = restrictions[uid][type] is True

    return {"restricted": restricted}


@router.get("/user/{user_id}/result")
def get_user_result(user_id: int):

    try:
        users = fetch_features()
        results = predict_batch(users)

        for row in results:
            if row["user_id"] == user_id:

                result = {
                    "user_id": user_id,
                    "risk_score": row["risk_score"],
                    "risk_level": row["risk_level"],
                    "timestamp": datetime.utcnow().isoformat()
                }

                save_history(result)
                return result

    except Exception as e:
        return {"error": str(e)}

    return {"message": "User not found"}


@router.get("/user/{user_id}/explanation")
def get_user_explanation(user_id: int):

    try:
        users = fetch_features()

        for u in users:
            if u["user_id"] == user_id:
                result = predict_user(u)

                return {
                    "user_id": user_id,
                    "risk_level": result["risk_level"],
                    "reasons": result["top_reasons"],
                    "actions": result["actions"]
                }

        return {"message": "Explanation not found"}

    except Exception as e:
        return {"error": str(e)}


@router.get("/users")
def get_all_users():
    try:
        users = fetch_features()
        results = predict_batch(users)
        return results
    except Exception as e:
        return {"error": str(e)}


@router.get("/user/{user_id}/restrictions")
def get_user_restrictions(user_id: int):
    os.makedirs(RUNTIME_PATH, exist_ok=True)

    if not os.path.exists(RESTRICTIONS_FILE):
        with open(RESTRICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}

    try:
        with open(RESTRICTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    return data.get(str(user_id), {})


@router.get("/user/{user_id}/history")
def get_user_history(user_id: int):
    os.makedirs(RUNTIME_PATH, exist_ok=True)

    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        history = []

    user_history = [h for h in history if h["user_id"] == user_id]

    user_history = user_history[-50:]

    if len(user_history) > 20:
        step = len(user_history) // 20
        user_history = user_history[::step]

    return user_history


@router.post("/analyze_user")
def analyze_user(req: AnalyzeUserRequest):

    user_id = req.user_id

    try:
        users = fetch_features()

        target_user = next((u for u in users if u["user_id"] == user_id), None)

        if not target_user:
            return {"status": "error", "message": "User not found"}

        result = predict_user(target_user)

        restrictions = load_restrictions()
        user_restrictions = restrictions.get(str(user_id), {})

        return {
            "status": "success",
            "data": {
                "user_id": user_id,
                "risk_score": result.get("risk_score"),
                "risk_level": result.get("risk_level"),
                "reasons": result.get("top_reasons", []),
                "actions": result.get("actions", []),
                "restrictions": user_restrictions
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}