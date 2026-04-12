# core/inference.py

import mysql.connector
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Support multiple import contexts (monolith app, component package, legacy flat path).
try:
    from ...models.sbert_model import get_model
except ImportError:
    try:
        from components.recommendation.models.sbert_model import get_model
    except ImportError:
        from models.sbert_model import get_model

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "ossn",
}


model = get_model()

CATEGORIES = {
        "Memes": ["meme", "funny", "joke"],
        "Food": ["food", "eat", "recipe", "restaurant", "cooking"],
        "Finance": ["money", "finance", "investment", "bank", "stock"],
        "Travel": ["travel", "trip", "hotel", "beach", "tour"],
        "Health": ["health", "doctor", "medical", "diet"],
        "Books": ["book", "reading", "novel", "author"],
        "Fitness": ["fitness", "gym", "workout", "exercise"],
        "Fashion": ["fashion", "style", "clothes", "dress"],
        "Music": ["music", "song", "band", "artist"],
        "Education": ["education", "study", "school", "university"],
        "Technology": ["technology", "tech", "ai", "software", "computer"],
        "Art": ["art", "drawing", "painting"],
        "Pets": ["pet", "dog", "cat", "animal"],
        "Gaming": ["game", "gaming", "playstation", "xbox"],
        "Photography": ["photo", "camera", "photography"],
        "Politics": ["politics", "government", "election"]
    }

def extract_categories(text):
    text = text.lower()
    matched = set()

    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                matched.add(category)
                break

    return matched

def run_sbert_recommendation():
    model = get_model()
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
        title = (row['title'] or "").strip()
        description = (row['description'] or "").strip()

        # Prefer title (your dataset uses it as category)
        text = title if title else description
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
    for i, user in enumerate(user_ids):

        # delete ONLY this user's old recs
        cur.execute("""
            DELETE FROM ossn_ng_friend_recs 
            WHERE user_guid = %s AND model='SBERT'
        """, (user,))

        sims = list(enumerate(sim_matrix[i]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

        user_text = texts[i]
        user_categories = extract_categories(user_text)

        for j, score in sims[1:6]:
            rec_user = user_ids[j]

            rec_text = texts[j]
            rec_categories = extract_categories(rec_text)

            shared = user_categories.intersection(rec_categories)
            shared_interests = ", ".join(list(shared)[:3]) if shared else "General"

            similarity_percentage = round(float(score) * 100, 2)

            cur.execute("""
                INSERT INTO ossn_ng_friend_recs
                (user_guid, rec_guid, model, shared_interests, similarity_score, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (user, rec_user, "SBERT", shared_interests, similarity_percentage))

    conn.commit()
    cur.close()
    conn.close()

    return {"status": "success"}