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
    EfficientNet-B2 based model for facial emotion recognition.
    """

    def __init__(self, num_classes=8, dropout_rate=0.4, pretrained=False):
        super().__init__()

        # Backbone: EfficientNet-B2
        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        self.num_features = self.backbone.num_features  # 1408 for B2

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
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
    'dropout_rate': 0.4
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

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="AffectNet Emotion Recognition API",
    description="Deep learning-based facial emotion recognition using EfficientNet-B2",
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
        
        # Load trained weights
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        checkpoint_path = os.path.join(base_dir, 'models', 'affectnet', 'affectnet_emotion_model_weights.pth')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
        "backbone": "EfficientNet-B2",
        "num_classes": MODEL_CONFIG['num_classes'],
        "image_size": MODEL_CONFIG['image_size'],
        "total_parameters": total_params,
        "device": str(device),
        "emotions": list(EMOTION_LABELS.values())
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded image
    
    Args:
        file: Uploaded image file (jpg, jpeg, png)
        
    Returns:
        JSON response with emotion prediction and confidence scores
    """
    
    # Check if model is loaded
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later."
        )
    
    # Validate file type
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (jpg, jpeg, png)"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess
        image_tensor = preprocess_image(image)
        
        # Predict
        results = predict_emotion(image_tensor)
        
        # Add metadata
        results['metadata'] = {
            'filename': file.filename,
            'image_size': image.size,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {results['predicted_emotion']} (confidence: {results['confidence']:.4f})")
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict emotions from multiple uploaded images
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON response with predictions for all images
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
            # Validate file type
            if file.content_type and not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid file type',
                    'success': False
                })
                continue
            
            # Read and process image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_tensor = preprocess_image(image)
            
            # Predict
            prediction = predict_emotion(image_tensor)
            prediction['filename'] = file.filename
            prediction['success'] = True
            
            results.append(prediction)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                'filename': file.filename,
                'error': str(e),
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
    
    # Run the application (port 8001 to avoid conflict with emo-app on 8000)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
