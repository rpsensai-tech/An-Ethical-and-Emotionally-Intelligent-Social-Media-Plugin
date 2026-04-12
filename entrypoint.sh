#!/bin/bash

# If the Azure Storage is mounted to /app/mounted_models, we create symlinks
# so the application expects the models exactly where they used to be downloaded.
if [ -d "/app/mounted_models" ]; then
    echo "[INFO] Azure Blob Storage mount detected at /app/mounted_models. Creating symlinks..."
    
    # Remove existing empty models directories if they exist, then link
    # Behavior component is at: /app/components/behavior/behavior_detection_component/models
    rm -rf /app/components/behavior/behavior_detection_component/models
    rm -rf /app/components/cyberbullying/models
    rm -rf /app/components/cyberbullying/assets
    rm -rf /app/components/emotion/models
    rm -rf /app/components/recommendation/model_config
    
    # Create required parent directories just in case
    mkdir -p /app/components/behavior/behavior_detection_component
    mkdir -p /app/components/cyberbullying
    mkdir -p /app/components/emotion
    mkdir -p /app/components/recommendation

    # Link the folders to match where the code expects models
    ln -s /app/mounted_models/behavior/models /app/components/behavior/behavior_detection_component/models
    ln -s /app/mounted_models/cyberbullying/models /app/components/cyberbullying/models
    ln -s /app/mounted_models/cyberbullying/assets /app/components/cyberbullying/assets
    ln -s /app/mounted_models/emotion/models /app/components/emotion/models
    
    echo "[INFO] Symlinks created successfully."
else
    echo "[WARN] /app/mounted_models not found. Proceeding without Azure mount."
fi

# Start the application
echo "[INFO] Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
