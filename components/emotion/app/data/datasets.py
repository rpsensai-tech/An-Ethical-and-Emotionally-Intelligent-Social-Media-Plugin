"""
Complete Dataset Loaders for GoEmotions and FER2013
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging


class GoEmotionsDataset(Dataset):
    """GoEmotions text emotion dataset loader"""
    
    def __init__(self, data_dir: Path, split: str = "train", tokenizer=None, max_length: int = 128):
        """
        Args:
            data_dir: Path to GoEmotions data directory
            split: 'train', 'dev', or 'test'
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        # Load emotion labels
        emotions_file = self.data_dir / "emotions.txt"
        if emotions_file.exists():
            with open(emotions_file, 'r') as f:
                self.emotions = [line.strip() for line in f if line.strip()]
        else:
            # Default GoEmotions 28 emotions
            self.emotions = [
                'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                'relief', 'remorse', 'sadness', 'surprise', 'neutral'
            ]
        
        # Load data
        self.data = self._load_data()
        self.logger.info(f"Loaded {len(self.data)} samples from {split} split")
    
    def _load_data(self) -> List[Dict]:
        """Load TSV data file"""
        file_map = {'train': 'train.tsv', 'dev': 'dev.tsv', 'test': 'test.tsv'}
        file_path = self.data_dir / file_map.get(self.split, 'train.tsv')
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    text = parts[0]
                    # Parse emotion labels (comma-separated indices)
                    emotion_indices = [int(idx) for idx in parts[1].split(',') if idx.strip().isdigit()]
                    
                    # Create multi-hot label vector
                    labels = np.zeros(len(self.emotions), dtype=np.float32)
                    for idx in emotion_indices:
                        if 0 <= idx < len(self.emotions):
                            labels[idx] = 1.0
                    
                    data.append({
                        'text': text,
                        'labels': labels,
                        'emotion_ids': emotion_indices
                    })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        if self.tokenizer is not None:
            # Tokenize text
            encoding = self.tokenizer(
                item['text'],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(item['labels'], dtype=torch.float32)
            }
        else:
            return {
                'text': item['text'],
                'labels': torch.tensor(item['labels'], dtype=torch.float32)
            }


class FER2013Dataset(Dataset):
    """FER2013 facial emotion dataset loader"""
    
    def __init__(self, data_dir: Path, split: str = "train", transform=None):
        """
        Args:
            data_dir: Path to FER2013 data directory
            split: 'train' or 'test'
            transform: Image transformations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        
        # FER2013 emotions
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        self.logger.info(f"Loaded {len(self.samples)} images from {split} split")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load image paths and labels"""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        samples = []
        for emotion in self.emotions:
            emotion_dir = split_dir / emotion
            if not emotion_dir.exists():
                self.logger.warning(f"Emotion directory not found: {emotion_dir}")
                continue
            
            # Load all images
            for img_path in emotion_dir.glob("*.jpg"):
                samples.append((img_path, self.emotion_to_idx[emotion]))
            for img_path in emotion_dir.glob("*.png"):
                samples.append((img_path, self.emotion_to_idx[emotion]))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert label to one-hot
        label_tensor = torch.zeros(len(self.emotions), dtype=torch.float32)
        label_tensor[label] = 1.0
        
        return image, label_tensor


class CustomEmotionDataset(Dataset):
    """Generic custom emotion dataset loader"""
    
    def __init__(self, data_file: Path, emotions: List[str], tokenizer=None, 
                 max_length: int = 128, is_multi_label: bool = True):
        """
        Args:
            data_file: Path to CSV/JSON file with 'text' and 'emotions' columns
            emotions: List of emotion labels
            tokenizer: Optional tokenizer for text
            max_length: Max sequence length
            is_multi_label: Whether dataset has multi-label annotations
        """
        self.data_file = Path(data_file)
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multi_label = is_multi_label
        self.emotion_to_idx = {e: i for i, e in enumerate(emotions)}
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load data from file"""
        if self.data_file.suffix == '.csv':
            df = pd.read_csv(self.data_file)
        elif self.data_file.suffix == '.json':
            df = pd.read_json(self.data_file)
        else:
            raise ValueError(f"Unsupported file format: {self.data_file.suffix}")
        
        data = []
        for _, row in df.iterrows():
            text = row['text']
            
            if self.is_multi_label:
                # Parse multi-label emotions
                if isinstance(row['emotions'], str):
                    emotion_list = [e.strip() for e in row['emotions'].split(',')]
                else:
                    emotion_list = row['emotions']
                
                labels = np.zeros(len(self.emotions), dtype=np.float32)
                for emotion in emotion_list:
                    if emotion in self.emotion_to_idx:
                        labels[self.emotion_to_idx[emotion]] = 1.0
            else:
                # Single label
                emotion = row['emotion']
                labels = np.zeros(len(self.emotions), dtype=np.float32)
                if emotion in self.emotion_to_idx:
                    labels[self.emotion_to_idx[emotion]] = 1.0
            
            data.append({'text': text, 'labels': labels})
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                item['text'],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(item['labels'], dtype=torch.float32)
            }
        else:
            return {
                'text': item['text'],
                'labels': torch.tensor(item['labels'], dtype=torch.float32)
            }


# Factory function
def create_dataset(dataset_type: str, data_dir: Path, split: str = "train", **kwargs):
    """Factory function to create appropriate dataset"""
    if dataset_type == "goemotions":
        return GoEmotionsDataset(data_dir, split, **kwargs)
    elif dataset_type == "fer2013":
        return FER2013Dataset(data_dir, split, **kwargs)
    elif dataset_type == "custom":
        return CustomEmotionDataset(data_dir / f"{split}.csv", **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
