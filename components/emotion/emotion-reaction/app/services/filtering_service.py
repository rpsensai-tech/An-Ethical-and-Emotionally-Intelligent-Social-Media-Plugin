"""
Ethical Filtering Service
Proactive filtering of harmful, toxic, or unethical content
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import logging


class EthicalFilteringService:
    """
    Proactive filtering service for harmful content detection
    
    This service implements:
    - Keyword-based filtering for offensive content
    - Pattern matching for harmful queries
    - Toxicity scoring
    - Content safety classification
    """
    
    def __init__(self, toxicity_threshold: float = 0.7):
        self.toxicity_threshold = toxicity_threshold
        self.logger = logging.getLogger(__name__)
        
        # Define harmful keyword categories
        self.harmful_keywords = self._load_harmful_keywords()
        self.harmful_patterns = self._load_harmful_patterns()
        
        self.logger.info("EthicalFilteringService initialized")
    
    def _load_harmful_keywords(self) -> Dict[str, List[str]]:
        """Load harmful keyword categories"""
        return {
            "violence": [
                "kill", "murder", "assault", "attack", "harm", "hurt", "abuse",
                "torture", "weapon", "gun", "knife", "bomb", "shoot"
            ],
            "harassment": [
                "bully", "harass", "threaten", "stalk", "intimidate", "blackmail"
            ],
            "hate_speech": [
                "hate", "racist", "sexist", "discriminate", "bigot", "slur"
            ],
            "self_harm": [
                "suicide", "self-harm", "cut myself", "end my life", "kill myself"
            ],
            "sexual_explicit": [
                "nsfw", "explicit", "pornographic", "sexual content"
            ],
            "illegal_activity": [
                "drugs", "steal", "hack", "fraud", "scam", "illegal"
            ]
        }
    
    def _load_harmful_patterns(self) -> List[Dict[str, Any]]:
        """Load regex patterns for harmful content"""
        return [
            {
                "pattern": r"how\s+to\s+(kill|harm|hurt|attack)",
                "category": "violence",
                "severity": "high"
            },
            {
                "pattern": r"(ways|methods)\s+to\s+(die|suicide)",
                "category": "self_harm",
                "severity": "critical"
            },
            {
                "pattern": r"(hate|attack|target)\s+(people|person|group)",
                "category": "hate_speech",
                "severity": "high"
            },
            {
                "pattern": r"(find|get|buy)\s+(weapons|guns|drugs)",
                "category": "illegal_activity",
                "severity": "high"
            }
        ]
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze content for harmful elements
        
        Args:
            text: Content to analyze
        
        Returns:
            Dictionary with safety analysis results
        """
        
        text_lower = text.lower()
        
        # Keyword matching
        keyword_matches = self._match_keywords(text_lower)
        
        # Pattern matching
        pattern_matches = self._match_patterns(text_lower)
        
        # Calculate toxicity score
        toxicity_score = self._calculate_toxicity_score(
            keyword_matches, pattern_matches
        )
        
        # Determine if content should be blocked
        is_harmful = toxicity_score >= self.toxicity_threshold
        
        # Determine severity level
        severity = self._determine_severity(toxicity_score, pattern_matches)
        
        # Generate explanation
        explanation = self._generate_explanation(
            is_harmful, keyword_matches, pattern_matches, severity
        )
        
        return {
            "is_harmful": is_harmful,
            "should_block": is_harmful,
            "toxicity_score": round(toxicity_score, 4),
            "severity": severity,
            "categories_detected": list(keyword_matches.keys()),
            "keyword_matches": keyword_matches,
            "pattern_matches": pattern_matches,
            "explanation": explanation,
            "recommendation": self._get_recommendation(is_harmful, severity)
        }
    
    def filter_search_query(self, query: str) -> Dict[str, Any]:
        """
        Filter search queries for harmful content
        
        Args:
            query: Search query to filter
        
        Returns:
            Dictionary with filtering decision and reasoning
        """
        
        analysis = self.analyze_content(query)
        
        return {
            "query": query,
            "allowed": not analysis["is_harmful"],
            "blocked": analysis["is_harmful"],
            "reason": analysis["explanation"],
            "severity": analysis["severity"],
            "toxicity_score": analysis["toxicity_score"],
            "alternative_suggestion": self._suggest_alternative(query) if analysis["is_harmful"] else None
        }
    
    def batch_filter(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Filter multiple texts"""
        return [self.analyze_content(text) for text in texts]
    
    def _match_keywords(self, text: str) -> Dict[str, List[str]]:
        """Match harmful keywords in text"""
        matches = {}
        
        for category, keywords in self.harmful_keywords.items():
            found = [kw for kw in keywords if kw in text]
            if found:
                matches[category] = found
        
        return matches
    
    def _match_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Match harmful patterns in text"""
        matches = []
        
        for pattern_dict in self.harmful_patterns:
            pattern = pattern_dict["pattern"]
            if re.search(pattern, text, re.IGNORECASE):
                matches.append({
                    "category": pattern_dict["category"],
                    "severity": pattern_dict["severity"],
                    "pattern": pattern
                })
        
        return matches
    
    def _calculate_toxicity_score(self, keyword_matches: Dict[str, List[str]],
                                   pattern_matches: List[Dict[str, Any]]) -> float:
        """Calculate overall toxicity score (0.0 to 1.0)"""
        
        # Base score from keyword matches
        keyword_score = min(len(keyword_matches) * 0.15, 0.6)
        
        # Pattern match scores
        pattern_score = 0.0
        for match in pattern_matches:
            if match["severity"] == "critical":
                pattern_score += 0.4
            elif match["severity"] == "high":
                pattern_score += 0.3
            else:
                pattern_score += 0.2
        
        pattern_score = min(pattern_score, 0.8)
        
        # Combined score
        total_score = min(keyword_score + pattern_score, 1.0)
        
        return total_score
    
    def _determine_severity(self, toxicity_score: float, 
                           pattern_matches: List[Dict[str, Any]]) -> str:
        """Determine severity level"""
        
        # Check for critical patterns
        for match in pattern_matches:
            if match["severity"] == "critical":
                return "critical"
        
        if toxicity_score >= 0.9:
            return "critical"
        elif toxicity_score >= 0.7:
            return "high"
        elif toxicity_score >= 0.5:
            return "medium"
        elif toxicity_score >= 0.3:
            return "low"
        else:
            return "safe"
    
    def _generate_explanation(self, is_harmful: bool,
                             keyword_matches: Dict[str, List[str]],
                             pattern_matches: List[Dict[str, Any]],
                             severity: str) -> str:
        """Generate human-readable explanation"""
        
        if not is_harmful:
            return "Content appears safe and appropriate."
        
        categories = list(keyword_matches.keys())
        
        if severity == "critical":
            return (f"Content blocked due to critical safety concerns. "
                   f"Detected categories: {', '.join(categories)}. "
                   f"This content may promote harm or illegal activity.")
        elif severity == "high":
            return (f"Content blocked due to high-risk elements. "
                   f"Detected: {', '.join(categories)}. "
                   f"This content may be harmful or offensive.")
        else:
            return (f"Content flagged for review. "
                   f"Detected potentially concerning elements: {', '.join(categories)}.")
    
    def _get_recommendation(self, is_harmful: bool, severity: str) -> str:
        """Get recommendation for content handling"""
        
        if not is_harmful:
            return "allow"
        
        if severity == "critical":
            return "block_immediately"
        elif severity == "high":
            return "block_and_report"
        elif severity == "medium":
            return "flag_for_review"
        else:
            return "monitor"
    
    def _suggest_alternative(self, query: str) -> str:
        """Suggest alternative phrasing for blocked queries"""
        
        # Simple suggestions based on context
        if "how to" in query.lower():
            return "Try rephrasing your search to focus on educational or informational content."
        
        return "Please rephrase your search using appropriate language."
    
    def add_custom_keywords(self, category: str, keywords: List[str]):
        """Add custom keywords to filtering"""
        if category not in self.harmful_keywords:
            self.harmful_keywords[category] = []
        self.harmful_keywords[category].extend(keywords)
        self.logger.info(f"Added {len(keywords)} keywords to category '{category}'")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        return {
            "total_keyword_categories": len(self.harmful_keywords),
            "total_keywords": sum(len(kw) for kw in self.harmful_keywords.values()),
            "total_patterns": len(self.harmful_patterns),
            "toxicity_threshold": self.toxicity_threshold
        }
