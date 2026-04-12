"""
Evaluation and Metrics Utilities
Comprehensive metrics for emotion classification models
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from typing import Dict, List, Any, Optional
import logging


class EmotionMetrics:
    """Calculate and track metrics for emotion classification"""
    
    def __init__(self, emotions: List[str], multi_label: bool = True):
        """
        Args:
            emotions: List of emotion labels
            multi_label: Whether this is multi-label classification
        """
        self.emotions = emotions
        self.multi_label = multi_label
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels (N, num_emotions) for multi-label or (N,) for single-label
            y_pred: Predicted labels (same shape as y_true)
            y_proba: Prediction probabilities (optional, for AUC metrics)
        
        Returns:
            Dictionary of metrics
        """
        if self.multi_label:
            return self._calculate_multilabel_metrics(y_true, y_pred, y_proba)
        else:
            return self._calculate_multiclass_metrics(y_true, y_pred, y_proba)
    
    def _calculate_multiclass_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate metrics for multi-class classification"""
        
        # Convert to 1D if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'per_class': {}
        }
        
        # Per-class metrics
        for i, emotion in enumerate(self.emotions):
            metrics['per_class'][emotion] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        # AUC metrics if probabilities provided
        if y_proba is not None:
            try:
                # One-hot encode true labels
                y_true_onehot = np.eye(len(self.emotions))[y_true]
                
                # ROC AUC (weighted)
                roc_auc_weighted = roc_auc_score(
                    y_true_onehot, y_proba, average='weighted', multi_class='ovr'
                )
                metrics['roc_auc_weighted'] = float(roc_auc_weighted)
                
                # Per-class AUC
                for i, emotion in enumerate(self.emotions):
                    try:
                        auc = roc_auc_score(y_true_onehot[:, i], y_proba[:, i])
                        metrics['per_class'][emotion]['roc_auc'] = float(auc)
                    except:
                        pass
            
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC metrics: {e}")
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()
        
        return metrics
    
    def _calculate_multilabel_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate metrics for multi-label classification"""
        
        # Subset accuracy (exact match)
        subset_accuracy = accuracy_score(y_true, y_pred)
        
        # Hamming loss (proportion of incorrect labels)
        hamming_loss = np.mean(y_true != y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Sample-based metrics
        precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
            y_true, y_pred, average='samples', zero_division=0
        )
        
        # Micro averages (aggregate contributions of all classes)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        # Macro averages (unweighted mean)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics = {
            'subset_accuracy': float(subset_accuracy),
            'hamming_loss': float(hamming_loss),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_samples': float(precision_samples),
            'recall_samples': float(recall_samples),
            'f1_samples': float(f1_samples),
            'per_class': {}
        }
        
        # Per-class metrics
        for i, emotion in enumerate(self.emotions):
            metrics['per_class'][emotion] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # AUC metrics if probabilities provided
        if y_proba is not None:
            try:
                # ROC AUC (macro average)
                roc_auc_macro = roc_auc_score(y_true, y_proba, average='macro')
                metrics['roc_auc_macro'] = float(roc_auc_macro)
                
                # Average Precision (macro)
                ap_macro = average_precision_score(y_true, y_proba, average='macro')
                metrics['average_precision_macro'] = float(ap_macro)
                
                # Per-class AUC and AP
                for i, emotion in enumerate(self.emotions):
                    try:
                        auc = roc_auc_score(y_true[:, i], y_proba[:, i])
                        ap = average_precision_score(y_true[:, i], y_proba[:, i])
                        metrics['per_class'][emotion]['roc_auc'] = float(auc)
                        metrics['per_class'][emotion]['average_precision'] = float(ap)
                    except:
                        pass
            
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC metrics: {e}")
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, Any], title: str = "Evaluation Metrics"):
        """Pretty print metrics"""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        
        if self.multi_label:
            print(f"\nOverall Metrics:")
            print(f"  Subset Accuracy:  {metrics['subset_accuracy']:.4f}")
            print(f"  Hamming Loss:     {metrics['hamming_loss']:.4f}")
            print(f"  F1 (Micro):       {metrics['f1_micro']:.4f}")
            print(f"  F1 (Macro):       {metrics['f1_macro']:.4f}")
            print(f"  F1 (Samples):     {metrics['f1_samples']:.4f}")
            
            if 'roc_auc_macro' in metrics:
                print(f"  ROC AUC (Macro):  {metrics['roc_auc_macro']:.4f}")
                print(f"  Avg Precision:    {metrics['average_precision_macro']:.4f}")
        else:
            print(f"\nOverall Metrics:")
            print(f"  Accuracy:         {metrics['accuracy']:.4f}")
            print(f"  Precision (Wgt):  {metrics['precision_weighted']:.4f}")
            print(f"  Recall (Wgt):     {metrics['recall_weighted']:.4f}")
            print(f"  F1 (Weighted):    {metrics['f1_weighted']:.4f}")
            print(f"  F1 (Macro):       {metrics['f1_macro']:.4f}")
            
            if 'roc_auc_weighted' in metrics:
                print(f"  ROC AUC (Wgt):    {metrics['roc_auc_weighted']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 70)
        
        for emotion, class_metrics in metrics['per_class'].items():
            print(f"{emotion:<15} "
                  f"{class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} "
                  f"{class_metrics['f1']:<12.4f} "
                  f"{class_metrics['support']:<10}")
        
        print("=" * 70 + "\n")
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str):
        """Save metrics to JSON file"""
        import json
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {output_file}")


def evaluate_model(model, dataloader, device: torch.device, 
                  emotions: List[str], multi_label: bool = True) -> Dict[str, Any]:
    """
    Evaluate a PyTorch model on a dataloader
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        emotions: List of emotion labels
        multi_label: Whether this is multi-label classification
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract inputs and labels
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
            else:
                inputs, labels = batch
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
            
            # Forward pass
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            else:
                outputs = model(inputs)
            
            # Get probabilities
            if multi_label:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                # Convert to one-hot for consistency
                preds_onehot = torch.zeros_like(probs)
                preds_onehot.scatter_(1, preds.unsqueeze(1), 1)
                preds = preds_onehot
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_proba = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    metrics_calculator = EmotionMetrics(emotions, multi_label)
    metrics = metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)
    
    return metrics
