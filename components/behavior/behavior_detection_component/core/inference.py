import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ============================================================
# BehaviourGuard API-READY ENGINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# -------------------------------
# Load Models (ONLY ONCE)
# -------------------------------
early_xgb = joblib.load(os.path.join(MODELS_DIR, "early_xgb_model.pkl"))
early_scaler = joblib.load(os.path.join(MODELS_DIR, "early_scaler.pkl"))

xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_behavior_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "behavior_scaler.pkl"))

iso = joblib.load(os.path.join(MODELS_DIR, "iso_model.pkl"))
anom_scaler = joblib.load(os.path.join(MODELS_DIR, "anom_scaler.pkl"))

with open(os.path.join(MODELS_DIR, "evidence_refs.json"), "r") as f:
    refs = json.load(f)

with open(os.path.join(MODELS_DIR, "population_baseline.json"), "r") as f:
    pop_baseline = json.load(f)


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
        return abs(value - base["median"]) / (base["iqr"] + 1e-9)

    scores = []
    for c in FEATURE_COLS:
        scores.append((c, deviation(row[c], baseline[c])))

    scores.sort(key=lambda x: x[1], reverse=True)

    reasons = []
    for feat, score in scores:
        if score < dev_threshold:
            continue
        direction = "high" if row[feat] >= baseline[feat]["median"] else "low"
        reasons.append(f"{feat} is unusually {direction}")
        if len(reasons) >= top_k:
            break

    return reasons


# -------------------------------
# Moderation Suggestions
# -------------------------------
def generate_actions(result):
    actions = []
    seen = set()

    risk = result.get("risk_level", "")
    reasons = result.get("top_reasons", [])

    # 🔴 Risk-based rule
    if risk in ["HIGH", "CRITICAL"]:
        actions.append({
            "action": "LIMIT_POSTING",
            "reason": "High risk behaviour detected",
            "enforceable": True
        })
        seen.add("LIMIT_POSTING")

    # 🧠 Reason-based rules
    for reason in reasons:

        if "inter_event" in reason and "LIMIT_POSTING" not in seen:
            actions.append({
                "action": "LIMIT_POSTING",
                "reason": reason,
                "enforceable": True
            })
            seen.add("LIMIT_POSTING")

        if "activity_rate" in reason and "LIMIT_COMMENTS" not in seen:
            actions.append({
                "action": "LIMIT_COMMENTS",
                "reason": reason,
                "enforceable": True
            })
            seen.add("LIMIT_COMMENTS")

    return actions


# ============================================================
# 🔥 CORE FUNCTION (MAIN TRANSFORMATION)
# ============================================================

def predict_user(user_dict: dict):

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

    # -------------------------------
    # Actions
    # -------------------------------
    result_temp = {
        "risk_level": level,
        "top_reasons": reasons
    }

    actions = generate_actions(result_temp)

    # -------------------------------
    # FINAL OUTPUT
    # -------------------------------
    return {
        "user_id": int(user_dict.get("user_id", -1)),
        "risk_score": round(score, 2),
        "risk_level": level,
        "evidence_score": round(ev, 3),
        "top_reasons": reasons,
        "actions": actions,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# -------------------------------
# Batch support
# -------------------------------
def predict_batch(users: list):
    return [predict_user(u) for u in users]