import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from azure_downloader import download_blob_if_not_exists
from pathlib import Path

# ============================================================
# BehaviourGuard API-READY ENGINE
# ============================================================

# Use Path for better path management
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Define local paths for all models and assets
EARLY_XGB_PATH = MODELS_DIR / "early_xgb_model.pkl"
EARLY_SCALER_PATH = MODELS_DIR / "early_scaler.pkl"
XGB_PATH = MODELS_DIR / "xgb_behavior_model.pkl"
SCALER_PATH = MODELS_DIR / "behavior_scaler.pkl"
ISO_PATH = MODELS_DIR / "iso_model.pkl"
ANOM_SCALER_PATH = MODELS_DIR / "anom_scaler.pkl"
REFS_PATH = MODELS_DIR / "evidence_refs.json"
BASELINE_PATH = MODELS_DIR / "population_baseline.json"

def download_behavior_assets():
    """Downloads all models and assets for the behavior component."""
    print("[INFO] Checking for behavior models and assets...")
    # Models
    download_blob_if_not_exists("behavior/models/early_xgb_model.pkl", EARLY_XGB_PATH)
    download_blob_if_not_exists("behavior/models/early_scaler.pkl", EARLY_SCALER_PATH)
    download_blob_if_not_exists("behavior/models/xgb_behavior_model.pkl", XGB_PATH)
    download_blob_if_not_exists("behavior/models/behavior_scaler.pkl", SCALER_PATH)
    download_blob_if_not_exists("behavior/models/iso_model.pkl", ISO_PATH)
    download_blob_if_not_exists("behavior/models/anom_scaler.pkl", ANOM_SCALER_PATH)
    # Assets
    download_blob_if_not_exists("behavior/models/evidence_refs.json", REFS_PATH)
    download_blob_if_not_exists("behavior/models/population_baseline.json", BASELINE_PATH)
    print("[INFO] Behavior assets check complete.")

# Trigger the download on module load
download_behavior_assets()


# -------------------------------
# Load Models (ONLY ONCE)
# -------------------------------
early_xgb = joblib.load(EARLY_XGB_PATH)
early_scaler = joblib.load(EARLY_SCALER_PATH)

xgb = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)

iso = joblib.load(ISO_PATH)
anom_scaler = joblib.load(ANOM_SCALER_PATH)

with open(REFS_PATH, "r") as f:
    refs = json.load(f)

with open(BASELINE_PATH, "r") as f:
    pop_baseline = json.load(f)

# -------------------------------
# Simple in-memory cache
# -------------------------------
USER_CACHE = {}
CACHE_TTL_SECONDS = 60  # you can change (e.g., 30–120)


# -------------------------------
# Feature columns
# -------------------------------
FEATURE_COLS = [
    "account_age_days",
    "activity_duration_days",
    "activity_rate",
    "mean_inter_event_minutes",
    "std_inter_event_minutes",
    "inter_event_cv",
    "burst_index",
]

MIN_EVENTS_REQUIRED = 5

FEATURE_LABELS = {
    "account_age_days": "account age",
    "activity_duration_days": "activity duration",
    "activity_rate": "activity rate",
    "mean_inter_event_minutes": "average time between actions",
    "std_inter_event_minutes": "timing variability",
    "inter_event_cv": "interaction irregularity",
    "burst_index": "burst activity",
}

REASON_TEMPLATES = {
    "activity_rate_high": "Unusually high activity rate compared with normal users",
    "activity_rate_low": "Unusually low activity rate compared with normal users",
    "mean_inter_event_minutes_low": "Actions are occurring unusually quickly",
    "mean_inter_event_minutes_high": "Long gaps between actions compared with typical behaviour",
    "std_inter_event_minutes_high": "Action timing is highly variable",
    "inter_event_cv_high": "Interaction timing is unusually irregular",
    "burst_index_high": "Behaviour shows bursty concentrated activity",
    "account_age_days_low": "Account is relatively new compared with typical users",
    "activity_duration_days_low": "Behaviour history is still limited",
}


def _baseline_stat(feat: str, key: str, default=None):
    feat_base = pop_baseline.get(feat, {})
    return feat_base.get(key, default)


def _reason_from_feature(feat: str, value: float) -> str | None:
    median = _baseline_stat(feat, "median")
    q90 = _baseline_stat(feat, "q90")
    q95 = _baseline_stat(feat, "q95")

    if median is None:
        return None

    if feat == "activity_rate":
        if q95 is not None and value >= q95:
            return REASON_TEMPLATES["activity_rate_high"]
        if value < median:
            return REASON_TEMPLATES["activity_rate_low"]

    if feat == "mean_inter_event_minutes":
        if value < median:
            return REASON_TEMPLATES["mean_inter_event_minutes_low"]
        if q95 is not None and value >= q95:
            return REASON_TEMPLATES["mean_inter_event_minutes_high"]

    if feat == "std_inter_event_minutes":
        if q90 is not None and value >= q90:
            return REASON_TEMPLATES["std_inter_event_minutes_high"]

    if feat == "inter_event_cv":
        if q90 is not None and value >= q90:
            return REASON_TEMPLATES["inter_event_cv_high"]

    if feat == "burst_index":
        if q90 is not None and value >= q90:
            return REASON_TEMPLATES["burst_index_high"]

    if feat == "account_age_days":
        if value < median:
            return REASON_TEMPLATES["account_age_days_low"]

    if feat == "activity_duration_days":
        if value < median:
            return REASON_TEMPLATES["activity_duration_days_low"]

    return None


# -------------------------------
# Risk level
# -------------------------------
def risk_level_from_score(score: float) -> str:
    if pd.isna(score):
        return "INSUFFICIENT_DATA"
    if score >= 85:
        return "CRITICAL"
    elif score >= 70:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    else:
        return "LOW"


# -------------------------------
# Evidence Score
# -------------------------------
def compute_evidence_from_events(df: pd.DataFrame, refs: dict) -> np.ndarray:
    T_ref = float(refs["T_ref"])
    D_ref = float(refs["D_ref"])
    A_ref = float(refs["A_ref"])

    if "total_events" in df.columns:
        total_events = df["total_events"].astype(float).values
    else:
        total_events = (df["activity_rate"].astype(float) *
                        df["activity_duration_days"].astype(float)).values
        total_events = np.nan_to_num(total_events, nan=0.0)

    tweet_score = np.minimum(total_events / (T_ref + 1e-6), 1.0)
    duration_score = np.minimum(df["activity_duration_days"].values / (D_ref + 1e-6), 1.0)
    age_score = np.minimum(df["account_age_days"].values / (A_ref + 1e-6), 1.0)

    return (tweet_score + duration_score + age_score) / 3.0


# -------------------------------
# Dynamic alpha
# -------------------------------
def dynamic_alpha(p_sup: np.ndarray, alpha_min=0.5, alpha_max=0.85):
    conf = np.abs(p_sup - 0.5) * 2.0
    return alpha_min + (alpha_max - alpha_min) * conf


# -------------------------------
# Explainability
# -------------------------------
def explain_user(row, baseline, top_k=3, dev_threshold=1.5):
    def deviation(value, base):
        median = base.get("median", 0.0)
        iqr = base.get("iqr", 1.0)
        return abs(value - median) / (iqr + 1e-9)

    scored = []
    for feat in FEATURE_COLS:
        feat_base = baseline.get(feat, {})
        score = deviation(row[feat], feat_base)
        scored.append((feat, score, row[feat]))

    scored.sort(key=lambda x: x[1], reverse=True)

    reasons = []
    seen = set()

    for feat, score, value in scored:
        if score < dev_threshold:
            continue

        reason = _reason_from_feature(feat, value)

        if not reason:
            direction = "high" if value >= baseline.get(feat, {}).get("median", 0) else "low"
            pretty = FEATURE_LABELS.get(feat, feat.replace("_", " "))
            reason = f"Unusual {pretty} ({direction} compared with normal users)"

        if reason not in seen:
            reasons.append(reason)
            seen.add(reason)

        if len(reasons) >= top_k:
            break

    return reasons


# -------------------------------
# Moderation Suggestions
# -------------------------------
# -------------------------------
# Moderation Suggestions
# -------------------------------
def generate_actions(row, risk_level, reasons):
    actions = []
    seen = set()

    def get_q(feat, key):
        return pop_baseline.get(feat, {}).get(key)

    activity_rate = float(row.get("activity_rate", 0))
    inter_event_cv = float(row.get("inter_event_cv", 0))
    mean_gap = float(row.get("mean_inter_event_minutes", 0))
    burst_index = float(row.get("burst_index", 0))
    account_age = float(row.get("account_age_days", 0))

    q95_activity = get_q("activity_rate", "q95")
    q90_cv = get_q("inter_event_cv", "q90")
    q95_gap = get_q("mean_inter_event_minutes", "q95")
    q90_burst = get_q("burst_index", "q90")

    # 1. High activity spike
    if q95_activity is not None and activity_rate >= q95_activity and "LIMIT_POSTING" not in seen:
        actions.append({
            "action": "LIMIT_POSTING",
            "reason": "User activity rate is extremely high compared with normal users",
            "enforceable": True
        })
        seen.add("LIMIT_POSTING")

    # 2. Highly irregular timing
    if q90_cv is not None and inter_event_cv >= q90_cv and "LIMIT_COMMENTS" not in seen:
        actions.append({
            "action": "LIMIT_COMMENTS",
            "reason": "Interaction timing is highly irregular",
            "enforceable": True
        })
        seen.add("LIMIT_COMMENTS")

    # 3. Bursty concentrated behaviour
    if q90_burst is not None and burst_index >= q90_burst and "LIMIT_POSTING" not in seen:
        actions.append({
            "action": "LIMIT_POSTING",
            "reason": "Behaviour shows unusually bursty activity",
            "enforceable": True
        })
        seen.add("LIMIT_POSTING")

    # 4. Unusual long action gaps
    if q95_gap is not None and mean_gap >= q95_gap and "LIMIT_INTERACTIONS" not in seen:
        actions.append({
            "action": "LIMIT_INTERACTIONS",
            "reason": "Unusual delay pattern detected between actions",
            "enforceable": False
        })
        seen.add("LIMIT_INTERACTIONS")

    # 5. High-risk new account
    if risk_level in ["HIGH", "CRITICAL"] and account_age < 7 and "REQUIRE_CAPTCHA" not in seen:
        actions.append({
            "action": "REQUIRE_CAPTCHA",
            "reason": "High-risk behaviour detected from a newly created account",
            "enforceable": False
        })
        seen.add("REQUIRE_CAPTCHA")

    # 6. Critical fallback escalation
    if risk_level == "CRITICAL" and "TEMP_SUSPEND" not in seen:
        actions.append({
            "action": "TEMP_SUSPEND",
            "reason": "Critical risk level indicates strong bot-like behaviour",
            "enforceable": False
        })
        seen.add("TEMP_SUSPEND")

    # 7. Reason-based support when statistical rules do not fire
    for reason in reasons:
        if "activity rate" in reason.lower() and "LIMIT_COMMENTS" not in seen:
            actions.append({
                "action": "LIMIT_COMMENTS",
                "reason": reason,
                "enforceable": True
            })
            seen.add("LIMIT_COMMENTS")

        if "irregular" in reason.lower() and "LIMIT_POSTING" not in seen:
            actions.append({
                "action": "LIMIT_POSTING",
                "reason": reason,
                "enforceable": True
            })
            seen.add("LIMIT_POSTING")

        if len(actions) >= 3:
            break

    return actions[:3]

# ============================================================
#  CORE FUNCTION (MAIN TRANSFORMATION)
# ============================================================

def predict_user(user_dict: dict):

    user_id = int(user_dict.get("user_id", -1))
    now = datetime.now(timezone.utc)

    # -------------------------------
    # Cache check
    # -------------------------------
    if user_id in USER_CACHE:
        cached = USER_CACHE[user_id]
        age = (now - cached["timestamp"]).total_seconds()

        if age < CACHE_TTL_SECONDS:
            return cached["data"]

    df = pd.DataFrame([user_dict])

    # Ensure features exist
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    # -------------------------------
    # Evidence
    # -------------------------------
    evidence = compute_evidence_from_events(df, refs)

    # -------------------------------
    # Early risk
    # -------------------------------
    X_early = df[FEATURE_COLS]
    X_early_scaled = early_scaler.transform(X_early)

    p_early = early_xgb.predict_proba(X_early_scaled)[:, 1]
    early_risk = np.minimum(100 * p_early, 75)

    # -------------------------------
    # Mature risk
    # -------------------------------
    X_mature = scaler.transform(df[FEATURE_COLS])

    p_sup = xgb.predict_proba(X_mature)[:, 1]

    anom_raw = -iso.decision_function(df[FEATURE_COLS])
    anom = anom_scaler.transform(anom_raw.reshape(-1, 1)).ravel()
    anom = np.clip(anom, 0, 1)

    alpha = dynamic_alpha(p_sup)
    p_hybrid = alpha * p_sup + (1 - alpha) * anom
    mature_risk = 100 * p_hybrid

    # -------------------------------
    # Fusion
    # -------------------------------
    conf = np.abs(p_hybrid - 0.5) * 2
    weight = np.power(evidence, 2) + (1 - np.power(evidence, 2)) * (0.7 * conf)
    weight = np.clip(weight, 0, 1)

    risk_final = weight * mature_risk + (1 - weight) * early_risk

    score = float(risk_final[0])
    ev = float(evidence[0])

    # -------------------------------
    # Risk level
    # -------------------------------
    level = risk_level_from_score(score)

    # -------------------------------
    # Explainability
    # -------------------------------
    if score < 50:
        reasons = ["Behavior within normal range"]
    else:
        reasons = explain_user(df.iloc[0], pop_baseline)

    # Strong fallback if nothing detected
    if not reasons:
        reasons = []

        for feat in FEATURE_COLS:
            value = df.iloc[0][feat]
            median = pop_baseline.get(feat, {}).get("median", None)

            if median is None:
                continue

            direction = "higher" if value >= median else "lower"
            pretty = FEATURE_LABELS.get(feat, feat.replace("_", " "))

            reasons.append(f"{pretty.capitalize()} is {direction} than typical users")

            if len(reasons) >= 2:
                break

        # Final safety fallback
        if not reasons:
            reasons = ["Unusual behavioural pattern detected compared with normal users"]

    # -------------------------------
    # Actions
    # -------------------------------
    row_dict = df.iloc[0].to_dict()
    if level == "LOW":
        actions = []
    else:
        actions = generate_actions(row_dict, level, reasons)

        if not actions and level in ["HIGH", "CRITICAL"]:
            actions = [{
                "action": "MONITOR",
                "reason": "Suspicious behaviour detected, further observation is required",
                "enforceable": False
            }]

    # -------------------------------
    # FINAL OUTPUT
    # -------------------------------
    result = {
        "user_id": user_id,
        "risk_score": round(score, 2),
        "risk_level": level,
        "evidence_score": round(ev, 3),
        "top_reasons": reasons,
        "actions": actions,
        "timestamp": now.isoformat()
    }

    # -------------------------------
    # Save to cache
    # -------------------------------
    USER_CACHE[user_id] = {
        "data": result,
        "timestamp": now
    }

    return result


# -------------------------------
# Batch support
# -------------------------------
def predict_batch(users: list):
    return [predict_user(u) for u in users]