# Recommendation Component (SBERT-Based)

This component provides friend recommendations using a pretrained SBERT model integrated via a FastAPI backend.

The system analyzes user posts, computes semantic similarity between users, and returns:
- Recommended users
- Shared interest categories (explainability)
- Similarity score (percentage)

---

## How It Works

1. User posts are fetched from `ossn_object`
2. Text is converted into embeddings using SBERT
3. Cosine similarity is computed between users
4. Top matches are stored in the database
5. FastAPI serves recommendations to OSSN

---

## Project Structure
recommendation/
├── app/
│ ├── main.py # FastAPI app initialization
│ └── api/
│ └── routes.py # API routes (unified endpoint)
├── core/
│ └── inference.py # SBERT logic & DB update
├── requirements.txt
└── README.md

## Requirements

Install dependencies:
pip install -r requirements.txt

Recommended Python version:Python 3.10+

Run in the backend:
uvicorn app.main:app --reload

API documentation: http://127.0.0.1:8000/docs

## PI Endpoint
Unified Endpoint
POST /api

# Get Recommendations

Request:

{
  "component": "recommendation",
  "action": "get",
  "user_id": 123
}

Response:

{
  "status": "success",
  "data": [
    {
      "rec_guid": 5770090410794371000,
      "shared_interests": "Photography, Technology, Memes",
      "similarity_score": 87.81
    }
  ]
}
# Refresh Recommendations (Run SBERT)

This recomputes all recommendations and updates the database.

Request:

{
  "component": "recommendation",
  "action": "refresh"
}

Response:

{
  "status": "success"
}

# Important Notes
The FastAPI backend must be running for OSSN to fetch recommendations.
SBERT runs only when /refresh is triggered (not on every request).
Recommendations are stored in the database for fast retrieval.
Shared interests are mapped into predefined categories for explainability.
Cold-start users are handled using selected interests.



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


## System Flow
OSSN (PHP)
   ↓
FastAPI (/api)
   ↓
Database (recommendations)
   ↓
Displayed in UI


## Known Limitations
Requires backend server to be running
SBERT refresh is not real-time (manual trigger)


## Integration Notes
OSSN should call /api endpoint (NOT /recommend)
Ensure request format includes:
component
action
API must be accessible (local or deployed) before using the plugin