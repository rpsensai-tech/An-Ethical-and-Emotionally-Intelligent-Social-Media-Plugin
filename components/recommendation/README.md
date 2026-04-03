This component provides friend recommendations using SBERT.
Pretrained SBERT model is loaded dynamically.

Run:
uvicorn app.main:app --reload

Endpoints:
POST /recommend → get recommendations
POST /refresh → recompute recommendations

Requirements:
pip install -r requirements.txt


## Database Requirements

### Table: ossn_ng_friend_recs

Stores SBERT-based friend recommendations.

| Column             | Type        | Description |
|------------------|------------|------------|
| id               | bigint     | Primary key |
| user_guid        | bigint     | Current user |
| rec_guid         | bigint     | Recommended user |
| model            | varchar    | Model name (SBERT) |
| shared_interests | text       | Common categories |
| similarity_score | float      | Similarity percentage |
| created_at       | datetime   | Timestamp |


CREATE TABLE ossn_ng_friend_recs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_guid BIGINT,
    rec_guid BIGINT,
    model VARCHAR(50),
    shared_interests TEXT,
    similarity_score FLOAT,
    created_at DATETIME
);

---

### Table: ossn_cold_start_interests

Stores selected interests for new users.

| Column        | Type     | Description |
|--------------|---------|------------|
| id           | bigint  | Primary key |
| user_guid    | bigint  | User ID |
| topics       | text    | JSON list of interests |
| time_created | int     | Timestamp |



CREATE TABLE ossn_cold_start_interests (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_guid BIGINT UNIQUE,
    topics TEXT,
    time_created INT
);