"""
Enhanced Text Analysis Service
Combines emotion detection with sarcasm and slang detection for improved accuracy
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .sarcasm_detection_service import SarcasmDetectionService
from .slang_detection_service import SlangDetectionService


class EnhancedTextAnalysisService:
    """
    Unified service combining emotion, sarcasm, and slang detection.
    Provides comprehensive text analysis with contextual understanding.
    """
    
    def __init__(
        self,
        sarcasm_model_path: Optional[str] = None,
        slang_dict_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the enhanced text analysis service.
        
        Args:
            sarcasm_model_path: Path to sarcasm model
            slang_dict_path: Path to slang dictionary
            device: Computing device
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing EnhancedTextAnalysisService")
        
        # Initialize sub-services
        try:
            self.sarcasm_service = SarcasmDetectionService(
                model_path=sarcasm_model_path,
                device=device
            )
        except Exception as e:
            self.logger.warning(f"Sarcasm service not available: {e}")
            self.sarcasm_service = None
        
        try:
            self.slang_service = SlangDetectionService(dictionary_path=slang_dict_path)
        except Exception as e:
            self.logger.warning(f"Slang service not available: {e}")
            self.slang_service = None
        
        self.logger.info("✅ Enhanced text analysis service initialized")
    
    def analyze_comprehensive(
        self,
        text: str,
        emotion_result: Optional[Dict] = None,
        sarcasm_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform comprehensive text analysis including sarcasm and slang.
        
        Args:
            text: Input text
            emotion_result: Optional emotion prediction result
            sarcasm_threshold: Threshold for sarcasm detection
            
        Returns:
            Comprehensive analysis dictionary
        """
        analysis = {
            'text': text,
            'sarcasm': None,
            'slang': None,
            'recommendations': []
        }
        
        # Detect sarcasm
        if self.sarcasm_service:
            try:
                sarcasm_result = self.sarcasm_service.detect(text, threshold=sarcasm_threshold)
                analysis['sarcasm'] = {
                    'detected': sarcasm_result['is_sarcastic'],
                    'confidence': sarcasm_result['confidence'],
                    'probability': sarcasm_result['sarcasm_probability']
                }
                
                # Add recommendation if sarcasm detected
                if sarcasm_result['is_sarcastic']:
                    analysis['recommendations'].append({
                        'type': 'sarcasm',
                        'message': 'Sarcasm detected: Emotional polarity may be reversed',
                        'action': 'Consider inverting emotion prediction (positive ↔ negative)',
                        'confidence': sarcasm_result['confidence']
                    })
            except Exception as e:
                self.logger.error(f"Sarcasm detection failed: {e}")
                analysis['sarcasm'] = {'error': str(e)}
        
        # Detect slang
        if self.slang_service:
            try:
                slang_result = self.slang_service.detect(text)
                analysis['slang'] = {
                    'detected': slang_result['has_slang'],
                    'terms': slang_result['slang_terms'],
                    'definitions': slang_result['definitions'],
                    'count': slang_result['slang_count'],
                    'density': slang_result['slang_density']
                }
                
                # Add recommendations for slang
                if slang_result['has_slang']:
                    slang_terms_str = ', '.join(slang_result['slang_terms'][:3])
                    if len(slang_result['slang_terms']) > 3:
                        slang_terms_str += f" (+{len(slang_result['slang_terms']) - 3} more)"
                    
                    analysis['recommendations'].append({
                        'type': 'slang',
                        'message': f"Slang detected: {slang_terms_str}",
                        'action': 'Use slang definitions for better context understanding',
                        'terms': slang_result['slang_terms']
                    })
                
                # High informal language density
                if slang_result['slang_density'] > 0.3:
                    analysis['recommendations'].append({
                        'type': 'informal',
                        'message': 'High informal language density',
                        'action': 'Text contains significant casual/social media language',
                        'density': slang_result['slang_density']
                    })
            except Exception as e:
                self.logger.error(f"Slang detection failed: {e}")
                analysis['slang'] = {'error': str(e)}
        
        # If emotion result provided, adjust for sarcasm
        if emotion_result and analysis.get('sarcasm', {}).get('detected'):
            analysis['emotion_adjustment'] = self._suggest_emotion_adjustment(emotion_result)
        
        return analysis
    
    def _suggest_emotion_adjustment(self, emotion_result: Dict) -> Dict:
        """
        Suggest emotion adjustments based on sarcasm detection.
        
        Args:
            emotion_result: Original emotion prediction
            
        Returns:
            Suggested adjustments
        """
        top_emotion = emotion_result.get('top_emotion', '')
        
        if not top_emotion:
            return {'suggestion': 'No emotion to adjust'}
        
        # Get adjusted emotion
        if self.sarcasm_service:
            adjusted_emotion = self.sarcasm_service.adjust_emotion_for_sarcasm(
                top_emotion, 
                is_sarcastic=True
            )
            
            return {
                'original_emotion': top_emotion,
                'suggested_emotion': adjusted_emotion,
                'reason': 'Sarcasm detected - polarity reversal applied'
            }
        
        return {'suggestion': 'Sarcasm service not available'}
    
    def analyze_batch(
        self,
        texts: List[str],
        sarcasm_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts efficiently.
        
        Args:
            texts: List of input texts
            sarcasm_threshold: Threshold for sarcasm detection
            
        Returns:
            List of analysis results
        """
        results = []
        
        # Batch process sarcasm if available
        sarcasm_results = []
        if self.sarcasm_service:
            try:
                sarcasm_results = self.sarcasm_service.detect_batch(texts, threshold=sarcasm_threshold)
            except Exception as e:
                self.logger.error(f"Batch sarcasm detection failed: {e}")
        
        # Batch process slang if available
        slang_results = []
        if self.slang_service:
            try:
                slang_results = self.slang_service.detect_batch(texts)
            except Exception as e:
                self.logger.error(f"Batch slang detection failed: {e}")
        
        # Combine results
        for i, text in enumerate(texts):
            analysis = {
                'text': text,
                'sarcasm': None,
                'slang': None,
                'recommendations': []
            }
            
            # Add sarcasm results
            if i < len(sarcasm_results):
                sarcasm = sarcasm_results[i]
                analysis['sarcasm'] = {
                    'detected': sarcasm['is_sarcastic'],
                    'confidence': sarcasm['confidence'],
                    'probability': sarcasm['sarcasm_probability']
                }
                
                if sarcasm['is_sarcastic']:
                    analysis['recommendations'].append({
                        'type': 'sarcasm',
                        'message': 'Sarcasm detected',
                        'action': 'Reverse emotional polarity'
                    })
            
            # Add slang results
            if i < len(slang_results):
                slang = slang_results[i]
                analysis['slang'] = {
                    'detected': slang['has_slang'],
                    'terms': slang['slang_terms'],
                    'definitions': slang.get('definitions', {}),
                    'count': slang['slang_count'],
                    'density': slang['slang_density']
                }
                
                if slang['has_slang']:
                    analysis['recommendations'].append({
                        'type': 'slang',
                        'message': f"Slang: {', '.join(slang['slang_terms'][:3])}",
                        'action': 'Apply slang context'
                    })
            
            results.append(analysis)
        
        return results
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about available services."""
        info = {
            'sarcasm_service': 'not_available',
            'slang_service': 'not_available'
        }
        
        if self.sarcasm_service:
            info['sarcasm_service'] = self.sarcasm_service.get_model_info()
        
        if self.slang_service:
            info['slang_service'] = self.slang_service.get_dictionary_info()
        
        return info
