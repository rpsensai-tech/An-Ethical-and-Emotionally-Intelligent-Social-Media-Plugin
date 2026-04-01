# core/inference.py

import mysql.connector
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "ossn",
}

model = SentenceTransformer('all-MiniLM-L6-v2')


def run_sbert_recommendation():
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)

    # 1. Get posts
    cur.execute("""
        SELECT owner_guid, title, description
        FROM ossn_object
        WHERE type='object' AND subtype='post'
    """)
    rows = cur.fetchall()

    # 2. Build user text
    user_texts = defaultdict(list)

    for row in rows:
        text = f"{row['title']} {row['description']}".strip()
        if text:
            user_texts[row['owner_guid']].append(text)

    if not user_texts:
        return {"status": "no_data"}

    # 3. Combine text
    user_ids = list(user_texts.keys())
    texts = [" ".join(user_texts[u]) for u in user_ids]

    # 4. Embeddings
    embeddings = model.encode(texts)

    # 5. Similarity
    sim_matrix = cosine_similarity(embeddings)

    # 6. Clear old recs
    cur.execute("DELETE FROM ossn_ng_friend_recs WHERE model='SBERT'")

    # 7. Insert new recs
    for i, user in enumerate(user_ids):
        sims = list(enumerate(sim_matrix[i]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

        for j, score in sims[1:6]:
            rec_user = user_ids[j]

            cur.execute("""
                INSERT INTO ossn_ng_friend_recs
                (user_guid, rec_guid, model, shared_interests, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (user, rec_user, "SBERT", "semantic similarity"))

    conn.commit()
    cur.close()
    conn.close()

    return {"status": "success"}