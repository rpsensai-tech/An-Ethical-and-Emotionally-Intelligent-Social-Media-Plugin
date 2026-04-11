from datetime import datetime
import time

from assets.ossn_adapter import fetch_features
from core.inference import predict_batch
from app.api.routes import save_history


def run_full_scan():
    try:
        users = fetch_features()
        results = predict_batch(users)

        for r in results:
            save_history({
                "user_id": r["user_id"],
                "risk_score": r["risk_score"],
                "risk_level": r["risk_level"],
                "timestamp": datetime.utcnow().isoformat()
            })

        print(f" Scan completed for {len(results)} users")

    except Exception as e:
        print(" Scan error:", e)


def background_loop():
    while True:
        print(" Running background scan...")
        run_full_scan()
        time.sleep(1800)