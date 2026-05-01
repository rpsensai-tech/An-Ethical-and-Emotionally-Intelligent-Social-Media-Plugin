"""
FastAPI Application for AffectNet Emotion Recognition
Facial emotion recognition using EfficientNet-B2 deep learning model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import io
import cv2
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class EmotionRecognitionModel(nn.Module):
    """
    ConvNeXt-Base model for facial emotion recognition.
    Matches the architecture used during training on Google Colab.
    """

    def __init__(self, num_classes=8, dropout_rate=0.5, pretrained=False):
        super().__init__()

        # Backbone: ConvNeXt-Base pre-trained on ImageNet-22k
        self.backbone = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=0.2
        )

        self.num_features = self.backbone.num_features  # 1024 for ConvNeXt-Base

        # Classifier head (matches training notebook exactly)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Emotion labels mapping
EMOTION_LABELS = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'surprise',
    4: 'fear',
    5: 'disgust',
    6: 'anger',
    7: 'contempt'
}

# Model configuration
MODEL_CONFIG = {
    'num_classes': 8,
    'image_size': 224,
    'dropout_rate': 0.5
}

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
model_loaded = False

# Face detector — loaded once at startup (bundled with opencv-python, no extra files needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="AffectNet Emotion Recognition API",
    description="Deep learning-based facial emotion recognition using ConvNeXt-Base",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the trained emotion recognition model"""
    global model, model_loaded
    
    try:
        logger.info("Loading emotion recognition model...")
        
        # Initialize model
        model = EmotionRecognitionModel(
            num_classes=MODEL_CONFIG['num_classes'],
            dropout_rate=MODEL_CONFIG['dropout_rate'],
            pretrained=False
        )
        
        # Load trained weights with a safe fallback for older deployments
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        primary_checkpoint = os.path.join(
            base_dir,
            'models',
            'affectnet',
            'affectnet_emotion_model_production_best.pth'
        )
        fallback_checkpoint = os.path.join(
            base_dir,
            'models',
            'affectnet',
            'affectnet_emotion_model_weights.pth'
        )

        checkpoint_path = primary_checkpoint if os.path.exists(primary_checkpoint) else fallback_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.to(device)
        model.eval()
        
        model_loaded = True
        logger.info(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    load_model()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_and_crop_faces(image: Image.Image):
    """
    Detect human faces in the image using OpenCV Haar cascade.
    Returns a list of (face_crop_PIL, bbox_dict) tuples.
    An empty list means no face was found.
    """
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return []

    W, H = image.size
    results = []
    for (x, y, w, h) in faces:
        # Add 20% padding so the crop includes forehead/chin context
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        face_crop = image.crop((x1, y1, x2, y2))
        bbox = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        results.append((face_crop, bbox))

    return results


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model inference
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def predict_emotion(image_tensor: torch.Tensor) -> Dict:
    """
    Predict emotion from preprocessed image
    
    Args:
        image_tensor: Preprocessed image tensor
        
    Returns:
        Dictionary with prediction results
    """
    if not model_loaded:
        raise RuntimeError("Model not loaded")
    
    try:
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predictions
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to numpy for easier handling
            probs = probabilities.cpu().numpy()[0]
            predicted_idx = predicted_class.item()
            confidence_score = confidence.item()
        
        # Create results
        results = {
            'predicted_emotion': EMOTION_LABELS[predicted_idx],
            'confidence': float(confidence_score),
            'all_probabilities': {
                EMOTION_LABELS[i]: float(probs[i]) 
                for i in range(len(EMOTION_LABELS))
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AffectNet Emotion Recognition API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model_loaded,
        "device": str(device),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "emotions": "/emotions",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/emotions")
async def get_emotions():
    """Get list of supported emotions"""
    return {
        "emotions": list(EMOTION_LABELS.values()),
        "num_classes": len(EMOTION_LABELS),
        "labels_mapping": EMOTION_LABELS
    }


@app.get("/model-info")
async def get_model_info():
    """Get model configuration and statistics"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "model_name": "EmotionRecognitionModel",
        "backbone": "ConvNeXt-Base",
        "num_classes": MODEL_CONFIG['num_classes'],
        "image_size": MODEL_CONFIG['image_size'],
        "total_parameters": total_params,
        "device": str(device),
        "emotions": list(EMOTION_LABELS.values())
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Detect human faces in the uploaded image, then predict the emotion
    for each detected face.

    Args:
        file: Uploaded image file (jpg, jpeg, png)

    Returns:
        JSON response with per-face emotion predictions.
        Returns 422 if no human face is detected in the image.
    """

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, jpeg, png)"
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Step 1: Detect faces
        face_crops = detect_and_crop_faces(image)

        if len(face_crops) == 0:
            raise HTTPException(
                status_code=422,
                detail="No human face detected in the image. Please upload a clear photo of a face."
            )

        # Step 2: Run emotion model on each cropped face
        face_predictions = []
        for idx, (face_crop, bbox) in enumerate(face_crops):
            image_tensor = preprocess_image(face_crop)
            emotion = predict_emotion(image_tensor)
            face_predictions.append({
                "face_index": idx,
                "bbox": bbox,
                "predicted_emotion": emotion["predicted_emotion"],
                "confidence": emotion["confidence"],
                "all_probabilities": emotion["all_probabilities"]
            })

        logger.info(
            f"Detected {len(face_crops)} face(s): "
            f"{[p['predicted_emotion'] for p in face_predictions]}"
        )

        # Keep backward-compatible top-level fields for existing OSSN integration
        primary_face = max(face_predictions, key=lambda p: p['confidence'])

        return JSONResponse(content={
            "predicted_emotion": primary_face["predicted_emotion"],
            "confidence": primary_face["confidence"],
            "all_probabilities": primary_face["all_probabilities"],
            "faces_detected": len(face_crops),
            "faces": face_predictions,
            "metadata": {
                "filename": file.filename,
                "image_size": list(image.size),
                "timestamp": datetime.now().isoformat()
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Detect faces and predict emotions for multiple uploaded images.

    Args:
        files: List of uploaded image files

    Returns:
        JSON response with per-image, per-face predictions.
    """

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )

    results = []

    for file in files:
        try:
            if file.content_type and not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid file type',
                    'faces_detected': 0,
                    'success': False
                })
                continue

            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # Detect faces
            face_crops = detect_and_crop_faces(image)

            if len(face_crops) == 0:
                results.append({
                    'filename': file.filename,
                    'error': 'No human face detected in the image',
                    'faces_detected': 0,
                    'success': False
                })
                continue

            # Run emotion on each face
            face_predictions = []
            for idx, (face_crop, bbox) in enumerate(face_crops):
                image_tensor = preprocess_image(face_crop)
                emotion = predict_emotion(image_tensor)
                face_predictions.append({
                    'face_index': idx,
                    'bbox': bbox,
                    'predicted_emotion': emotion['predicted_emotion'],
                    'confidence': emotion['confidence'],
                    'all_probabilities': emotion['all_probabilities']
                })

            results.append({
                'filename': file.filename,
                'faces_detected': len(face_crops),
                'faces': face_predictions,
                'success': True
            })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                'filename': file.filename,
                'error': str(e),
                'faces_detected': 0,
                'success': False
            })

    return JSONResponse(content={
        'total_images': len(files),
        'successful': sum(1 for r in results if r.get('success', False)),
        'failed': sum(1 for r in results if not r.get('success', False)),
        'results': results,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )