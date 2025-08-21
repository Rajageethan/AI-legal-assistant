"""
Content filtering service with Guardrails integration
Provides additional layer of safety for legal chatbot responses
"""

import re
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

class ContentFilter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Harmful content categories
        self.harmful_categories = {
            'violence': [
                r'\b(kill|murder|assault|attack|violence|harm|hurt|weapon)\b',
                r'\b(fight|beat|punch|stab|shoot)\b'
            ],
            'illegal_activities': [
                r'\b(fraud|scam|money.laundering|tax.evasion)\b',
                r'\b(bribe|corruption|illegal.gambling)\b',
                r'\b(drug.trafficking|smuggling|counterfeiting)\b'
            ],
            'discrimination': [
                r'\b(racial|gender|religious|ethnic).*(discrimination|bias|prejudice)\b',
                r'\b(hate.speech|slur|offensive.language)\b'
            ],
            'privacy_violation': [
                r'\b(personal.information|social.security|credit.card)\b',
                r'\b(hack|breach|steal.data|identity.theft)\b'
            ]
        }
        
        # Legal misinformation patterns
        self.misinformation_patterns = [
            r'\b(law doesn\'t apply|above the law|legal immunity)\b',
            r'\b(always wins|never loses|guaranteed outcome)\b',
            r'\b(no consequences|risk.free|foolproof)\b',
            r'\b(loophole|workaround|bypass.law)\b'
        ]
        
        # Professional boundary patterns
        self.boundary_violations = [
            r'\b(i am your lawyer|attorney.client.privilege)\b',
            r'\b(legal.representation|represent you in court)\b',
            r'\b(file.lawsuit.for.you|handle.your.case)\b'
        ]

    def filter_user_input(self, message: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Filter user input for harmful or inappropriate content
        """
        result = {
            'is_filtered': False,
            'filter_reason': None,
            'severity': 'low',
            'categories': [],
            'filtered_message': message
        }
        
        # Check harmful content categories
        for category, patterns in self.harmful_categories.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    result['is_filtered'] = True
                    result['filter_reason'] = f"Contains {category} content"
                    result['severity'] = 'high'
                    result['categories'].append(category)
                    break
        
        # Check for boundary violations
        for pattern in self.boundary_violations:
            if re.search(pattern, message, re.IGNORECASE):
                result['is_filtered'] = True
                result['filter_reason'] = "Violates professional boundaries"
                result['severity'] = 'medium'
                result['categories'].append('boundary_violation')
                break
        
        # If filtered, provide alternative message
        if result['is_filtered']:
            result['filtered_message'] = self._generate_alternative_message(result['categories'])
        
        return result

    def filter_bot_response(self, response: str, sources: List[str]) -> Dict[str, Any]:
        """
        Filter bot response for misinformation and inappropriate content
        """
        result = {
            'is_filtered': False,
            'filter_reason': None,
            'severity': 'low',
            'issues': [],
            'filtered_response': response
        }
        
        # Check for misinformation patterns
        for pattern in self.misinformation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                result['is_filtered'] = True
                result['filter_reason'] = "Contains potential misinformation"
                result['severity'] = 'high'
                result['issues'].append('misinformation')
                break
        
        # Check for boundary violations in response
        for pattern in self.boundary_violations:
            if re.search(pattern, response, re.IGNORECASE):
                result['is_filtered'] = True
                result['filter_reason'] = "Response violates professional boundaries"
                result['severity'] = 'high'
                result['issues'].append('boundary_violation')
                break
        
        # Check for unsupported claims
        if self._contains_unsupported_claims(response, sources):
            result['is_filtered'] = True
            result['filter_reason'] = "Contains unsupported legal claims"
            result['severity'] = 'medium'
            result['issues'].append('unsupported_claims')
        
        # If filtered, generate corrected response
        if result['is_filtered']:
            result['filtered_response'] = self._correct_response(response, result['issues'])
        
        return result

    def _generate_alternative_message(self, categories: List[str]) -> str:
        """Generate alternative message for filtered content"""
        if 'violence' in categories:
            return "I understand you may be dealing with a difficult situation. I can help with legal information about personal safety, restraining orders, or reporting procedures."
        
        if 'illegal_activities' in categories:
            return "I can provide information about legal compliance and regulations, but cannot assist with activities that may violate the law."
        
        if 'boundary_violation' in categories:
            return "I'm an AI assistant that provides legal information, not a licensed attorney. I can help you understand legal concepts and direct you to appropriate resources."
        
        return "I can help with general legal information and guidance. Please rephrase your question to focus on understanding legal concepts or procedures."

    def _contains_unsupported_claims(self, response: str, sources: List[str]) -> bool:
        """Check if response contains claims not supported by sources"""
        # Strong claim indicators without proper attribution
        strong_claims = [
            r'\b(will definitely|guaranteed to|always results in)\b',
            r'\b(court will rule|judge will decide|you will win)\b',
            r'\b(no doubt|certainly|without question)\b'
        ]
        
        for pattern in strong_claims:
            if re.search(pattern, response, re.IGNORECASE):
                # Check if there's proper attribution nearby
                if not self._has_nearby_attribution(response, pattern):
                    return True
        
        return False

    def _has_nearby_attribution(self, response: str, claim_pattern: str) -> bool:
        """Check if there's attribution near a strong claim"""
        attribution_words = ['according to', 'based on', 'source', 'section', 'under']
        
        # Find claim position
        match = re.search(claim_pattern, response, re.IGNORECASE)
        if not match:
            return False
        
        # Check 100 characters before and after claim
        start = max(0, match.start() - 100)
        end = min(len(response), match.end() + 100)
        context = response[start:end].lower()
        
        return any(attr in context for attr in attribution_words)

    def _correct_response(self, response: str, issues: List[str]) -> str:
        """Correct filtered response"""
        corrected = response
        
        if 'misinformation' in issues:
            # Add qualifiers to absolute statements
            corrected = re.sub(
                r'\b(will definitely|guaranteed to|always results in)\b',
                r'may typically',
                corrected,
                flags=re.IGNORECASE
            )
            
            corrected = re.sub(
                r'\b(court will rule|judge will decide)\b',
                r'courts may rule',
                corrected,
                flags=re.IGNORECASE
            )
        
        if 'boundary_violation' in issues:
            # Remove attorney-client language
            corrected = re.sub(
                r'\b(i am your lawyer|attorney.client.privilege)\b',
                r'as an AI assistant providing information',
                corrected,
                flags=re.IGNORECASE
            )
        
        if 'unsupported_claims' in issues:
            # Add disclaimer for strong claims
            disclaimer = "\n\n*Note: Legal outcomes depend on specific circumstances and jurisdiction. This information is general in nature.*"
            corrected += disclaimer
        
        return corrected

class GuardrailsIntegration:
    """
    Integration with Guardrails AI for advanced content validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_available = self._check_guardrails_availability()
    
    def _check_guardrails_availability(self) -> bool:
        """Check if Guardrails is available and properly configured"""
        try:
            # Temporarily disable guardrails due to torch compatibility issues
            self.guardrails_available = False
            print("⚠️ Guardrails temporarily disabled - using basic content filtering only")
            return False
        except ImportError:
            self.guardrails_available = False
            print("⚠️ Guardrails not available - using basic content filtering only")
            return False
    
    async def validate_with_guardrails(self, text: str, validation_type: str = "legal_response") -> Dict[str, Any]:
        """
        Validate text using Guardrails if available
        """
        if not self.is_available:
            return {'validated': True, 'guardrails_used': False}
        
        try:
            # This would integrate with actual Guardrails validators
            # For now, return a mock response
            return {
                'validated': True,
                'guardrails_used': True,
                'validation_type': validation_type,
                'confidence': 0.95
            }
        except Exception as e:
            self.logger.error(f"Guardrails validation failed: {e}")
            return {'validated': True, 'guardrails_used': False, 'error': str(e)}

# Global instances
content_filter = ContentFilter()
guardrails_integration = GuardrailsIntegration()
