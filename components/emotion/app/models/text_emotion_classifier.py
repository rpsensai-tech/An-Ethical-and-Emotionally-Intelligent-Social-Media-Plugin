"""
Text Emotion Classifier
BERT/RoBERTa-based emotion classification for text
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np


class TextEmotionClassifier(nn.Module):
    """BERT/RoBERTa-based emotion classifier for multi-label emotion prediction"""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_emotions: int = 28, 
                 dropout_rate: float = 0.1):
        super(TextEmotionClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_emotions = num_emotions
        self.dropout_rate = dropout_rate
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_emotions)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict_emotions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                        threshold: float = 0.5) -> Dict[str, Any]:
        """Predict emotions with probabilities"""
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)
            
            return {
                'logits': logits.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'predictions': (probabilities > threshold).cpu().numpy()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'num_emotions': self.num_emotions,
            'dropout_rate': self.dropout_rate,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class TextModelManager:
    """Manage trained text emotion classification models"""
    
    def __init__(self, models_dir: Path, emotions: List[str]):
        self.models_dir = Path(models_dir)
        self.emotions = emotions
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Currently loaded model
        self.current_model: Optional[TextEmotionClassifier] = None
        self.current_tokenizer: Optional[AutoTokenizer] = None
        self.current_device: Optional[torch.device] = None
        self.current_model_name: Optional[str] = None
    
    def save_model(self, model: TextEmotionClassifier, tokenizer: AutoTokenizer, 
                   model_name: str, metrics: Dict[str, Any] = None, 
                   device: torch.device = None) -> str:
        """Save a trained model with metadata"""
        
        try:
            model_dir = self.models_dir / model_name
            self.logger.info(f"Creating model directory: {model_dir}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            model_path = model_dir / "model.pt"
            self.logger.info(f"Saving model state dict to: {model_path}")
            torch.save(model.state_dict(), model_path)
            
            if not model_path.exists():
                raise RuntimeError(f"Model file was not created: {model_path}")
            
            # Save tokenizer
            tokenizer_dir = model_dir / "tokenizer"
            self.logger.info(f"Saving tokenizer to: {tokenizer_dir}")
            tokenizer.save_pretrained(tokenizer_dir)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'base_model': model.model_name,
                'num_emotions': model.num_emotions,
                'dropout_rate': model.dropout_rate,
                'emotions': self.emotions,
                'device': str(device) if device else None,
                'metrics': metrics or {},
                'model_info': model.get_model_info(),
                'saved_at': str(datetime.now())
            }
            
            metadata_path = model_dir / "metadata.json"
            self.logger.info(f"Saving metadata to: {metadata_path}")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ Model saved successfully: {model_name}")
            self.logger.info(f"📁 Model directory: {model_dir}")
            self.logger.info(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model '{model_name}': {e}")
            raise RuntimeError(f"Model saving failed: {e}") from e
    
    def load_model(self, model_name: str, device: torch.device) -> Tuple[TextEmotionClassifier, AutoTokenizer]:
        """Load a saved model"""
        
        try:
            model_dir = self.models_dir / model_name
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create model
            model = TextEmotionClassifier(
                model_name=metadata['base_model'],
                num_emotions=metadata['num_emotions'],
                dropout_rate=metadata.get('dropout_rate', 0.1)  # Default to 0.1 if not specified
            )
            
            # Load weights
            model_path = model_dir / "model.pt"
            if not model_path.exists():
                model_path = model_dir / "model.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found in {model_dir}")
            
            # Temporarily add text-project to sys.path for loading models with dependencies
            import sys
            text_project_path = str(self.models_dir.parent.parent / "text-project")
            sys_path_backup = sys.path.copy()
            if text_project_path not in sys.path:
                sys.path.insert(0, text_project_path)
            
            try:
                # Load checkpoint (may contain nested structure)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            finally:
                # Restore sys.path
                sys.path = sys_path_backup
            
            # Extract state_dict if checkpoint is a dict with nested structure
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            
            # Load tokenizer
            tokenizer_dir = model_dir / "tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
            self.logger.info(f"✅ Model loaded: {model_name}")
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model '{model_name}': {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models = []
        
        if not self.models_dir.exists():
            return models
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        models.append({
                            'model_name': model_dir.name,
                            'base_model': metadata.get('base_model', 'unknown'),
                            'metrics': metadata.get('metrics', {}),
                            'saved_at': metadata.get('saved_at', 'unknown')
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to read metadata for {model_dir.name}: {e}")
        
        return models
    
    def predict_with_current_model(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Make prediction using currently loaded model"""
        
        if self.current_model is None or self.current_tokenizer is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Tokenize input
        inputs = self.current_tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.current_device) for k, v in inputs.items()}
        
        # Predict
        result = self.current_model.predict_emotions(
            inputs['input_ids'],
            inputs['attention_mask'],
            threshold=threshold
        )
        
        # Format probabilities
        emotion_probs = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotions, result['probabilities'][0])
        }
        
        return {
            'text': text,
            'emotion_probabilities': emotion_probs,
            'threshold': threshold
        }
