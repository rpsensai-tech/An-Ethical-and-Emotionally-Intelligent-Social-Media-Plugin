import os
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timezone

load_dotenv()

# ============================================================
# OSSN Adapter
# Produces features matching the trained BehaviourGuard feature set:
#   account_age_days
#   activity_duration_days
#   activity_rate
#   mean_inter_event_minutes
#   std_inter_event_minutes
#   inter_event_cv
#   burst_index
#
# Output: ossn_behavior_features.csv
# ============================================================


# ---------------------------
# CONFIG: Update if needed
# ---------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "ossn_db"),
}


# ---------------------------
# Helpers
# ---------------------------
def safe_to_datetime_from_unix(series: pd.Series) -> pd.Series:
    """Convert unix seconds (can be str/int/float) to pandas datetime (UTC)."""
    s = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(s, unit="s", utc=True, errors="coerce")


def compute_features_for_user(times_utc: pd.Series) -> dict:
    """
    Compute BehaviourGuard features from sorted timestamps (UTC).
    times_utc: pd.Series of datetime64[ns, UTC], already sorted.
    """

    n = len(times_utc)

    # If user has 0 events, return zeros (still allow scoring pipeline to run)
    if n == 0:
        return {
            "activity_duration_days": 0.0,
            "activity_rate": 0.0,
            "mean_inter_event_minutes": 0.0,
            "std_inter_event_minutes": 0.0,
            "inter_event_cv": 0.0,
            "burst_index": 0.0,
            "total_events": 0,
        }

    # If only 1 event, timing stats can't be computed; keep them 0
    # activity_duration_days is 0, activity_rate is 1 event/day over tiny window? -> keep stable:
    if n == 1:
        return {
            "activity_duration_days": 0.0,
            "activity_rate": 0.0,
            "mean_inter_event_minutes": 0.0,
            "std_inter_event_minutes": 0.0,
            "inter_event_cv": 0.0,
            "burst_index": 0.0,
            "total_events": 1,
        }

    # ----- activity_duration_days -----
    duration_days = (times_utc.iloc[-1] - times_utc.iloc[0]).total_seconds() / 86400.0
    # Match your research practice: +1 day in discrete days.
    # But here we keep it as continuous for better stability and same scaling.
    # Avoid divide-by-zero (min 10 minute in days).
    duration_days = max(duration_days, 10 / 1440)

    # ----- activity_rate (events/day) -----
    total_events = n
    activity_rate = total_events / duration_days

    # ----- inter-event features (minutes) -----
    diffs_min = times_utc.diff().dropna().dt.total_seconds() / 60.0
    mean_inter_event = float(diffs_min.mean())
    std_inter_event = float(diffs_min.std(ddof=0))  # stable for small samples
    inter_event_cv = float(std_inter_event / (mean_inter_event + 1e-6))

    # ----- burst_index (max hourly / mean hourly) -----
    hourly = times_utc.dt.floor("h")
    counts_per_hour = hourly.value_counts()

    max_hourly = float(counts_per_hour.max())
    mean_hourly = float(counts_per_hour.mean())
    burst_index = float(max_hourly / (mean_hourly + 1e-6))

    return {
        "activity_duration_days": float(duration_days),
        "activity_rate": float(activity_rate),
        "mean_inter_event_minutes": float(mean_inter_event),
        "std_inter_event_minutes": float(std_inter_event),
        "inter_event_cv": float(inter_event_cv),
        "burst_index": float(burst_index),
        "total_events": int(total_events),
    }


# ---------------------------
# MAIN
# ---------------------------
def main():
    # ---------- Connect ----------
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print(" Connected to OSSN database")
    except Exception as e:
        print(f" Database connection failed: {e}. Skipping feature fetch.")
        return pd.DataFrame()

    # ---------- Load users ----------
    # OSSN: ossn_users.guid (user id), ossn_users.time_created (unix seconds)
    users_query = """
        SELECT guid, time_created
        FROM ossn_users
    """
    users_df = pd.read_sql(users_query, conn)
    print("Users loaded:", len(users_df))

    # Convert user creation times to UTC datetime
    users_df["user_created_ts"] = safe_to_datetime_from_unix(users_df["time_created"])

    # ---------- Load posts ----------
    # OSSN posts are stored in ossn_object; use owner_guid + time_created
    # We avoid over-filtering because OSSN variants differ by type/subtype.
    posts_query = """
        SELECT owner_guid AS user_id, time_created
        FROM ossn_object
        WHERE type = 'user'
    """
    posts_df = pd.read_sql(posts_query, conn)
    posts_df["action"] = "post"

    # ---------- Load comments ----------
    comments_query = """
        SELECT owner_guid AS user_id, time_created
        FROM ossn_annotations
    """
    comments_df = pd.read_sql(comments_query, conn)
    comments_df["action"] = "comment"

    # ---------- LOAD LIKES ----------
    likes_query = """
    SELECT
        l.guid AS user_id,
        e.time_created
    FROM ossn_likes l
    JOIN ossn_entities e
    ON l.id = e.guid
    """

    likes_df = pd.read_sql(likes_query, conn)
    likes_df["action"] = "like"

    # ---------- Combine events ----------
    events = pd.concat([posts_df, comments_df, likes_df], ignore_index=True)
    print("Total raw events (posts+comments):", len(events))

    # Clean timestamps
    events["timestamp"] = safe_to_datetime_from_unix(events["time_created"])
    events = events.dropna(subset=["timestamp"])

    # Ensure numeric user_id
    events["user_id"] = pd.to_numeric(events["user_id"], errors="coerce")
    events = events.dropna(subset=["user_id"])
    events["user_id"] = events["user_id"].astype(int)

    events = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print("Total usable events:", len(events))

    # ---------- Compute per-user features ----------
    feature_rows = []

    for user_id, group in events.groupby("user_id"):
        times = group["timestamp"].sort_values().reset_index(drop=True)
        feat = compute_features_for_user(times)
        feat["user_id"] = int(user_id)
        feature_rows.append(feat)

    features_df = pd.DataFrame(feature_rows)

    # ---------- Merge account_age_days ----------
    # account_age_days = now_utc - user_created_ts
    now_utc = datetime.now(timezone.utc)
    users_df["account_age_days"] = (now_utc - users_df["user_created_ts"]).dt.total_seconds() / 86400.0

    features_df = features_df.merge(
        users_df[["guid", "account_age_days"]],
        left_on="user_id",
        right_on="guid",
        how="left"
    ).drop(columns=["guid"])

    # Fill missing (in case some events belong to non-user rows)
    features_df["account_age_days"] = features_df["account_age_days"].fillna(0.0)

    # ---------- Select ONLY trained feature columns (exact match) ----------
    # Your models expect these 7 feature columns:
    trained_feature_cols = [
        "account_age_days",
        "activity_duration_days",
        "activity_rate",
        "mean_inter_event_minutes",
        "std_inter_event_minutes",
        "inter_event_cv",
        "burst_index"
    ]

    # Ensure all exist
    for c in trained_feature_cols:
        if c not in features_df.columns:
            features_df[c] = 0.0

    # Final output for model scoring
    out_df = features_df[["user_id", "total_events"] + trained_feature_cols].copy()

    # Replace inf/nan
    out_df = out_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print("\n Final Feature Table (matches trained feature set)")
    print(out_df.head(20))


    # ---------- Close ----------
    conn.close()

    return out_df

def fetch_features():
    df = main()

    if df is None or df.empty:
        return []

    return df.to_dict(orient="records")


#if __name__ == "__main__":
#    main()