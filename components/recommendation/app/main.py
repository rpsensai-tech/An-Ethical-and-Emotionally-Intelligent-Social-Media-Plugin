# app/main.py

from fastapi import FastAPI
from core.inference import run_sbert_recommendation
import mysql.connector

app = FastAPI()

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
        SELECT rec_guid, shared_interests
        FROM ossn_ng_friend_recs
        WHERE user_guid=%s
        LIMIT 5
    """, (user_id,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {"recommendations": rows}


@app.post("/refresh")
def refresh():
    return run_sbert_recommendation()