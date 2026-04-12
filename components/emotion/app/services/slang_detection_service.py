"""
Slang Detection Service
Detects and interprets slang terms in text using dictionary-based approach
"""

import json
import re
from typing import Dict, List, Optional
import logging
from pathlib import Path


class SlangDetectionService:
    """
    Service for detecting slang terms in text.
    Uses dictionary-based matching with word boundary detection.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize the slang detection service.
        
        Args:
            dictionary_path: Path to slang dictionary JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing SlangDetectionService")
        
        # Load slang dictionary
        if dictionary_path is None:
            # Default to backend data directory
            dictionary_path = Path(__file__).parent.parent / "data" / "slang_dictionary.json"
        
        self.dictionary_path = Path(dictionary_path)
        self.slang_dict = self._load_dictionary()
        
        self.logger.info(f"✅ Slang dictionary loaded with {len(self.slang_dict)} terms")
    
    def _load_dictionary(self) -> Dict[str, str]:
        """Load slang dictionary from JSON file."""
        try:
            if not self.dictionary_path.exists():
                self.logger.warning(f"Slang dictionary not found at {self.dictionary_path}")
                return {}
            
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                slang_dict = json.load(f)
            
            return slang_dict
            
        except Exception as e:
            self.logger.error(f"Failed to load slang dictionary: {e}")
            return {}
    
    def detect_slang_terms(self, text: str) -> List[str]:
        """
        Detect slang terms in text using dictionary matching.
        Uses word boundary matching for accurate detection.
        
        Args:
            text: Input text
            
        Returns:
            List of detected slang terms
        """
        text_lower = text.lower()
        found_slang = []
        
        # Check each slang term in dictionary
        for slang in self.slang_dict.keys():
            # Use word boundary matching for accurate detection
            # This ensures we match whole words, not substrings
            pattern = r'\b' + re.escape(slang) + r'\b'
            if re.search(pattern, text_lower):
                found_slang.append(slang)
        
        return found_slang
    
    def detect(self, text: str) -> Dict:
        """
        Detect slang in a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with slang detection results
        """
        try:
            slang_terms = self.detect_slang_terms(text)
            has_slang = len(slang_terms) > 0
            
            # Get definitions for found slang
            definitions = {
                term: self.slang_dict.get(term, 'Unknown') 
                for term in slang_terms
            }
            
            # Calculate slang density
            words = text.split()
            word_count = len(words)
            slang_density = len(slang_terms) / word_count if word_count > 0 else 0
            
            return {
                'text': text,
                'has_slang': has_slang,
                'slang_detected': has_slang,
                'slang_terms': slang_terms,
                'definitions': definitions,
                'slang_count': len(slang_terms),
                'word_count': word_count,
                'slang_density': round(slang_density, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Slang detection failed: {e}")
            raise RuntimeError(f"Failed to detect slang: {e}") from e
    
    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """
        Detect slang in multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of detection results
        """
        results = []
        for text in texts:
            try:
                result = self.detect(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Slang detection failed for text: {e}")
                results.append({
                    'text': text,
                    'error': str(e),
                    'has_slang': False
                })
        return results
    
    def add_slang_term(self, term: str, definition: str):
        """
        Add a new slang term to the dictionary.
        
        Args:
            term: Slang term to add
            definition: Definition/meaning of the term
        """
        self.slang_dict[term.lower()] = definition
        self.logger.info(f"Added slang term: {term}")
    
    def update_dictionary(self, new_terms: Dict[str, str]):
        """
        Update dictionary with multiple terms.
        
        Args:
            new_terms: Dictionary of new slang terms and definitions
        """
        self.slang_dict.update({k.lower(): v for k, v in new_terms.items()})
        self.logger.info(f"Updated dictionary with {len(new_terms)} new terms")
    
    def save_dictionary(self):
        """Save the current dictionary to file."""
        try:
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(self.slang_dict, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Dictionary saved to {self.dictionary_path}")
        except Exception as e:
            self.logger.error(f"Failed to save dictionary: {e}")
    
    def get_definition(self, term: str) -> Optional[str]:
        """
        Get definition for a specific slang term.
        
        Args:
            term: Slang term to look up
            
        Returns:
            Definition or None if not found
        """
        return self.slang_dict.get(term.lower())
    
    def search_slang(self, query: str) -> List[Dict[str, str]]:
        """
        Search for slang terms matching a query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching terms with definitions
        """
        query_lower = query.lower()
        matches = []
        
        for term, definition in self.slang_dict.items():
            if query_lower in term or query_lower in definition.lower():
                matches.append({
                    'term': term,
                    'definition': definition
                })
        
        return matches
    
    def get_dictionary_info(self) -> Dict:
        """Get information about the slang dictionary."""
        return {
            "total_terms": len(self.slang_dict),
            "dictionary_path": str(self.dictionary_path),
            "status": "loaded" if self.slang_dict else "empty"
        }
    
    def get_all_terms(self) -> Dict[str, str]:
        """Get all slang terms and definitions."""
        return self.slang_dict.copy()
