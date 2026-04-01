This component provides friend recommendations using SBERT.
Pretrained SBERT model is loaded dynamically.

Run:
uvicorn app.main:app --reload

Endpoints:
POST /recommend → get recommendations
POST /refresh → recompute recommendations