"""
Safety and validation service for legal chatbot
Ensures user intentions are appropriate and bot responses are accurate and non-misleading
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum
import asyncio

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"

class IntentCategory(Enum):
    LEGAL_INFORMATION = "legal_information"
    LEGAL_ADVICE = "legal_advice"
    PROCEDURAL_GUIDANCE = "procedural_guidance"
    DOCUMENT_HELP = "document_help"
    INAPPROPRIATE = "inappropriate"
    HARMFUL = "harmful"

class SafetyService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Prohibited content patterns
        self.prohibited_patterns = [
            r'\b(hack|crack|illegal|fraud|scam|cheat)\b',
            r'\b(violence|harm|hurt|kill|murder)\b',
            r'\b(drugs|cocaine|heroin|marijuana)\b',
            r'\b(bomb|weapon|gun|explosive)\b',
            r'\b(suicide|self.harm|depression)\b'
        ]
        
        # Legal advice warning patterns
        self.legal_advice_patterns = [
            r'\bshould I (sue|file|claim|divorce)\b',
            r'\bwhat should I do (if|when|about)\b',
            r'\b(recommend|suggest|advise) (me|us) to\b',
            r'\bwill I (win|lose|succeed)\b',
            r'\bhow much (money|compensation|damages)\b'
        ]
        
        # Inappropriate legal patterns
        self.inappropriate_legal_patterns = [
            r'\bhow to (avoid|evade|escape) (taxes|law|legal)\b',
            r'\bhow to (hide|conceal|launder)\b',
            r'\b(loophole|workaround) (in|for) (law|legal)\b',
            r'\bhow to (lie|deceive|mislead) (in court|legally)\b'
        ]
        
        # Response quality indicators
        self.quality_indicators = [
            "according to",
            "based on",
            "generally",
            "typically",
            "may vary",
            "consult",
            "legal professional",
            "attorney",
            "lawyer"
        ]
        
        # Disclaimer requirements
        self.disclaimer_triggers = [
            "legal advice",
            "should do",
            "recommend",
            "suggest",
            "file a lawsuit",
            "sue",
            "legal action"
        ]

    async def validate_user_input(self, message: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive input validation for user messages
        Returns safety assessment and intent classification
        """
        
        # Basic sanitization
        message = message.strip().lower()
        
        # Check for prohibited content
        safety_check = self._check_prohibited_content(message)
        if safety_check['level'] == SafetyLevel.BLOCKED:
            return {
                'is_safe': False,
                'safety_level': SafetyLevel.BLOCKED,
                'reason': safety_check['reason'],
                'intent': IntentCategory.HARMFUL,
                'requires_disclaimer': False,
                'suggested_response': "I cannot provide assistance with that request. Please ask about legal information or procedures."
            }
        
        # Classify intent
        intent = self._classify_intent(message)
        
        # Check for inappropriate legal requests
        inappropriate_check = self._check_inappropriate_legal_content(message)
        if inappropriate_check['is_inappropriate']:
            return {
                'is_safe': False,
                'safety_level': SafetyLevel.BLOCKED,
                'reason': inappropriate_check['reason'],
                'intent': IntentCategory.INAPPROPRIATE,
                'requires_disclaimer': False,
                'suggested_response': "I cannot provide guidance on circumventing legal requirements. I can help with understanding legal processes and requirements."
            }
        
        # Check if request seeks legal advice
        advice_check = self._check_legal_advice_request(message)
        requires_disclaimer = advice_check['is_advice_request']
        
        # Determine safety level
        if intent == IntentCategory.LEGAL_ADVICE:
            safety_level = SafetyLevel.WARNING
        elif requires_disclaimer:
            safety_level = SafetyLevel.WARNING
        else:
            safety_level = SafetyLevel.SAFE
        
        return {
            'is_safe': True,
            'safety_level': safety_level,
            'intent': intent,
            'requires_disclaimer': requires_disclaimer,
            'advice_warning': advice_check.get('warning'),
            'content_flags': self._flag_content_concerns(message)
        }

    async def validate_bot_response(self, response: str, user_query: str, sources: List[str]) -> Dict[str, Any]:
        """
        Validate bot response for accuracy, appropriateness, and legal compliance
        """
        
        # Check response quality
        quality_score = self._assess_response_quality(response, sources)
        
        # Check for legal advice language
        advice_check = self._check_response_for_advice(response)
        
        # Verify source attribution
        source_check = self._verify_source_attribution(response, sources)
        
        # Check for misleading content
        misleading_check = self._check_misleading_content(response)
        
        # Check for required disclaimers
        disclaimer_check = self._check_disclaimers(response, user_query)
        
        # Determine if response needs modification
        needs_modification = (
            advice_check['contains_advice'] or
            misleading_check['is_misleading'] or
            disclaimer_check['needs_disclaimer'] or
            quality_score < 0.6
        )
        
        # Generate safety assessment
        safety_assessment = {
            'is_safe': not misleading_check['is_misleading'],
            'quality_score': quality_score,
            'needs_modification': needs_modification,
            'issues': [],
            'recommendations': []
        }
        
        # Collect issues and recommendations
        if advice_check['contains_advice']:
            safety_assessment['issues'].append("Response contains legal advice language")
            safety_assessment['recommendations'].append("Add disclaimer and soften advice language")
        
        if misleading_check['is_misleading']:
            safety_assessment['issues'].append(f"Potentially misleading: {misleading_check['reason']}")
            safety_assessment['recommendations'].append("Revise for accuracy and add qualifications")
        
        if disclaimer_check['needs_disclaimer']:
            safety_assessment['issues'].append("Missing required legal disclaimer")
            safety_assessment['recommendations'].append("Add appropriate legal disclaimer")
        
        if not source_check['properly_attributed']:
            safety_assessment['issues'].append("Poor source attribution")
            safety_assessment['recommendations'].append("Improve source citations")
        
        return safety_assessment

    def _check_prohibited_content(self, message: str) -> Dict[str, Any]:
        """Check for prohibited content patterns"""
        for pattern in self.prohibited_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    'level': SafetyLevel.BLOCKED,
                    'reason': f"Contains prohibited content: {pattern}"
                }
        return {'level': SafetyLevel.SAFE, 'reason': None}

    def _check_inappropriate_legal_content(self, message: str) -> Dict[str, Any]:
        """Check for inappropriate legal requests"""
        for pattern in self.inappropriate_legal_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    'is_inappropriate': True,
                    'reason': f"Request seeks to circumvent legal requirements"
                }
        return {'is_inappropriate': False, 'reason': None}

    def _classify_intent(self, message: str) -> IntentCategory:
        """Classify user intent based on message content"""
        
        # Legal advice indicators
        advice_indicators = [
            r'\bshould I\b', r'\bwhat do I do\b', r'\brecommend\b',
            r'\badvise\b', r'\bwill I win\b', r'\bhow much money\b'
        ]
        
        # Information request indicators
        info_indicators = [
            r'\bwhat is\b', r'\bwhat are\b', r'\bexplain\b',
            r'\bdefine\b', r'\bhow does\b', r'\btell me about\b'
        ]
        
        # Procedural guidance indicators
        procedure_indicators = [
            r'\bhow to file\b', r'\bwhat forms\b', r'\bwhat documents\b',
            r'\bwhat steps\b', r'\bprocess for\b', r'\bprocedure\b'
        ]
        
        for pattern in advice_indicators:
            if re.search(pattern, message, re.IGNORECASE):
                return IntentCategory.LEGAL_ADVICE
        
        for pattern in procedure_indicators:
            if re.search(pattern, message, re.IGNORECASE):
                return IntentCategory.PROCEDURAL_GUIDANCE
        
        for pattern in info_indicators:
            if re.search(pattern, message, re.IGNORECASE):
                return IntentCategory.LEGAL_INFORMATION
        
        return IntentCategory.LEGAL_INFORMATION

    def _check_legal_advice_request(self, message: str) -> Dict[str, Any]:
        """Check if message is requesting legal advice"""
        for pattern in self.legal_advice_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return {
                    'is_advice_request': True,
                    'warning': "This appears to be a request for legal advice. I can provide general information but cannot give specific legal advice."
                }
        return {'is_advice_request': False}

    def _flag_content_concerns(self, message: str) -> List[str]:
        """Flag potential content concerns"""
        flags = []
        
        if re.search(r'\b(emergency|urgent|immediate)\b', message, re.IGNORECASE):
            flags.append("urgent_legal_matter")
        
        if re.search(r'\b(court date|hearing|trial)\b', message, re.IGNORECASE):
            flags.append("active_legal_proceeding")
        
        if re.search(r'\b(arrest|charged|criminal)\b', message, re.IGNORECASE):
            flags.append("criminal_matter")
        
        return flags

    def _assess_response_quality(self, response: str, sources: List[str]) -> float:
        """Assess response quality based on various factors"""
        score = 1.0
        
        # Check for hedging language (good for legal responses)
        hedging_words = ['may', 'might', 'could', 'generally', 'typically', 'often']
        hedging_count = sum(1 for word in hedging_words if word in response.lower())
        hedging_score = min(hedging_count * 0.1, 0.3)
        
        # Check for source attribution
        source_score = 0.2 if sources and len(sources) > 0 else 0
        
        # Check for appropriate qualifications
        qualification_words = ['consult', 'attorney', 'lawyer', 'legal professional']
        qualification_score = 0.2 if any(word in response.lower() for word in qualification_words) else 0
        
        # Penalize absolute statements
        absolute_words = ['always', 'never', 'definitely', 'certainly', 'guaranteed']
        absolute_penalty = sum(0.1 for word in absolute_words if word in response.lower())
        
        final_score = score + hedging_score + source_score + qualification_score - absolute_penalty
        return max(0, min(1, final_score))

    def _check_response_for_advice(self, response: str) -> Dict[str, Any]:
        """Check if response contains legal advice language"""
        advice_phrases = [
            'you should', 'i recommend', 'i suggest', 'you must',
            'you need to', 'the best option is', 'i advise'
        ]
        
        for phrase in advice_phrases:
            if phrase in response.lower():
                return {
                    'contains_advice': True,
                    'advice_phrase': phrase
                }
        
        return {'contains_advice': False}

    def _verify_source_attribution(self, response: str, sources: List[str]) -> Dict[str, Any]:
        """Verify proper source attribution in response"""
        if not sources:
            return {
                'properly_attributed': len(response) < 100,  # Short responses may not need sources
                'has_sources': False
            }
        
        # Check if response mentions sources or legal authorities
        source_indicators = ['according to', 'based on', 'under', 'section', 'act']
        has_attribution = any(indicator in response.lower() for indicator in source_indicators)
        
        return {
            'properly_attributed': has_attribution,
            'has_sources': True,
            'source_count': len(sources)
        }

    def _check_misleading_content(self, response: str) -> Dict[str, Any]:
        """Check for potentially misleading content"""
        misleading_patterns = [
            r'\balways works\b',
            r'\bguaranteed (success|win|result)\b',
            r'\bno risk\b',
            r'\b100% (certain|sure|effective)\b',
            r'\bwill definitely\b'
        ]
        
        for pattern in misleading_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return {
                    'is_misleading': True,
                    'reason': f"Contains potentially misleading guarantee: {pattern}"
                }
        
        return {'is_misleading': False}

    def _check_disclaimers(self, response: str, user_query: str) -> Dict[str, Any]:
        """Check if response needs legal disclaimers"""
        needs_disclaimer = False
        
        # Check if user query or response triggers disclaimer requirement
        for trigger in self.disclaimer_triggers:
            if trigger in user_query.lower() or trigger in response.lower():
                needs_disclaimer = True
                break
        
        # Check if disclaimer already exists
        disclaimer_indicators = [
            'not legal advice', 'consult', 'attorney', 'lawyer',
            'legal professional', 'this information is general'
        ]
        
        has_disclaimer = any(indicator in response.lower() for indicator in disclaimer_indicators)
        
        return {
            'needs_disclaimer': needs_disclaimer and not has_disclaimer,
            'has_disclaimer': has_disclaimer
        }

    def generate_safety_enhanced_response(self, original_response: str, safety_assessment: Dict[str, Any], user_query: str) -> str:
        """Generate a safety-enhanced version of the response"""
        enhanced_response = original_response
        
        # Add disclaimers if needed
        if safety_assessment.get('needs_modification'):
            # Add legal disclaimer
            disclaimer = "\n\n⚠️ **Important**: This information is for general educational purposes only and does not constitute legal advice. Laws vary by jurisdiction and individual circumstances. For specific legal guidance, please consult with a qualified attorney."
            
            # Soften advice language
            enhanced_response = self._soften_advice_language(enhanced_response)
            
            # Add hedging language
            enhanced_response = self._add_hedging_language(enhanced_response)
            
            # Add disclaimer
            enhanced_response += disclaimer
        
        return enhanced_response

    def _soften_advice_language(self, response: str) -> str:
        """Replace advice language with softer alternatives"""
        replacements = {
            'you should': 'you may want to consider',
            'you must': 'it is typically required to',
            'you need to': 'it is generally advisable to',
            'i recommend': 'one option might be to',
            'i suggest': 'you might consider',
            'the best option is': 'one common approach is'
        }
        
        for original, replacement in replacements.items():
            response = re.sub(original, replacement, response, flags=re.IGNORECASE)
        
        return response

    def _add_hedging_language(self, response: str) -> str:
        """Add appropriate hedging language to legal responses"""
        # Add qualifiers to absolute statements
        response = re.sub(r'\b(This is|This means)\b', r'This generally means', response, flags=re.IGNORECASE)
        response = re.sub(r'\b(The law requires)\b', r'The law typically requires', response, flags=re.IGNORECASE)
        
        return response

    async def log_safety_event(self, event_type: str, details: Dict[str, Any]):
        """Log safety-related events for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        self.logger.info(f"Safety Event: {event_type}", extra=log_entry)

# Global safety service instance
safety_service = SafetyService()
