# app/main.py

from fastapi import FastAPI
from core.inference import run_sbert_recommendation
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "ossn",
}


@app.post("/recommend")
def recommend(data: dict):
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
        "recommendations": [
            {
                "rec_guid": row["rec_guid"],
                "shared_interests": row["shared_interests"],
                "similarity_score": row["similarity_score"]
            }
            for row in rows
        ]
    }


@app.post("/refresh")
def refresh():
    try:
        return run_sbert_recommendation()
    except Exception as e:
        return {"error": str(e)}