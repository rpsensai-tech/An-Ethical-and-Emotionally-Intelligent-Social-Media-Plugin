from fastapi import APIRouter
import mysql.connector

# Support multiple import contexts (monolith app, component package, legacy flat path).
try:
    from ...core.inference import run_sbert_recommendation
except ImportError:
    try:
        from components.recommendation.core.inference import run_sbert_recommendation
    except ImportError:
        from core.inference import run_sbert_recommendation

router = APIRouter()

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "ossn",
}


@router.post("/api")
def unified_api(data: dict):
    component = data.get("component")

    if not component:
        return {"error": "component is required"}

    # -----------------------------
    # RECOMMENDATION (YOUR PART)
    # -----------------------------
    if component == "recommendation":

        action = data.get("action")

        # 🔹 GET recommendations
        if action == "get":
            user_id = data.get("user_id")

            conn = mysql.connector.connect(**DB_CONFIG)
            cur = conn.cursor(dictionary=True)

            cur.execute("""
                SELECT rec_guid, shared_interests, similarity_score
                FROM ossn_ng_friend_recs
                WHERE user_guid=%s
                LIMIT 5
            """, (user_id,))

            rows = cur.fetchall()

            cur.close()
            conn.close()

            return {
                "status": "success",
                "data": [
                    {
                        "rec_guid": row["rec_guid"],
                        "shared_interests": row["shared_interests"],
                        "similarity_score": row["similarity_score"]
                    }
                    for row in rows
                ]
            }

        # 🔹 REFRESH SBERT
        elif action == "refresh":
            try:
                return run_sbert_recommendation()
            except Exception as e:
                return {"error": str(e)}

    # -----------------------------
    # UNKNOWN COMPONENT
    # -----------------------------
    return {"error": "Invalid component"}