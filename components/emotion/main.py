# -*- coding: utf-8 -*-
"""
Unified Entry Point for Emotion-Aware Social Media Platform
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("Unified Emotion Recognition API")
    print("=" * 70)
    print("Running a single FastAPI server...")
    print("Access points:")
    print("  * Primary API:       http://localhost:8000/api/v1")
    print("  * Primary Docs:      http://localhost:8000/docs")
    print("  * Image API:         http://localhost:8000/image-api")
    print("  * Image Docs:        http://localhost:8000/image-api/docs")
    print("=" * 70)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\nServer stopped.")

if __name__ == "__main__":
    main()
