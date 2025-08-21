"""
Safety configuration for legal chatbot
Centralized configuration for all safety and validation settings
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class SafetyMode(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"

@dataclass
class SafetyConfig:
    """Central configuration for safety settings"""
    
    # Global safety mode
    safety_mode: SafetyMode = SafetyMode.STANDARD
    
    # Input validation settings
    max_message_length: int = 500
    min_message_length: int = 1
    enable_profanity_filter: bool = True
    enable_intent_classification: bool = True
    
    # Content filtering settings
    block_harmful_content: bool = True
    filter_inappropriate_requests: bool = True
    detect_boundary_violations: bool = True
    
    # Response validation settings
    require_source_attribution: bool = True
    add_legal_disclaimers: bool = True
    soften_advice_language: bool = True
    minimum_quality_score: float = 0.6
    
    # Monitoring settings
    log_all_interactions: bool = True
    track_user_patterns: bool = True
    enable_real_time_alerts: bool = True
    
    # Rate limiting for safety
    max_violations_per_hour: int = 5
    max_violations_per_day: int = 20
    temporary_block_duration_minutes: int = 30
    
    # Guardrails integration
    enable_guardrails: bool = True
    guardrails_validation_level: str = "standard"

# Safety thresholds by mode
SAFETY_THRESHOLDS = {
    SafetyMode.BASIC: {
        'content_filter_sensitivity': 0.7,
        'response_modification_threshold': 0.8,
        'user_risk_threshold': 5.0,
        'require_disclaimers': False
    },
    SafetyMode.STANDARD: {
        'content_filter_sensitivity': 0.6,
        'response_modification_threshold': 0.6,
        'user_risk_threshold': 3.0,
        'require_disclaimers': True
    },
    SafetyMode.STRICT: {
        'content_filter_sensitivity': 0.4,
        'response_modification_threshold': 0.4,
        'user_risk_threshold': 2.0,
        'require_disclaimers': True
    }
}

# Legal-specific safety patterns
LEGAL_SAFETY_PATTERNS = {
    'prohibited_advice_requests': [
        r'\bhow to (avoid|evade|escape) (taxes|law|legal)\b',
        r'\bhow to (hide|conceal|launder) (money|assets)\b',
        r'\b(loophole|workaround) (in|for) (law|legal)\b',
        r'\bhow to (lie|deceive|mislead) (in court|legally)\b'
    ],
    'emergency_legal_situations': [
        r'\b(arrest|arrested|police|custody)\b',
        r'\b(court date|hearing|trial) (today|tomorrow)\b',
        r'\b(emergency|urgent|immediate) (legal|help)\b'
    ],
    'boundary_violations': [
        r'\bi am your (lawyer|attorney|client)\b',
        r'\b(attorney.client|lawyer.client) (privilege|relationship)\b',
        r'\b(represent me|be my lawyer|legal representation)\b'
    ]
}

# Response enhancement templates
DISCLAIMER_TEMPLATES = {
    'general_legal': "⚠️ **Important**: This information is for general educational purposes only and does not constitute legal advice. Laws vary by jurisdiction and individual circumstances. For specific legal guidance, please consult with a qualified attorney.",
    
    'emergency_situation': "🚨 **Emergency Legal Situation**: If this is an urgent legal matter, please contact a local attorney immediately or call your local bar association's referral service. This AI cannot provide immediate legal representation.",
    
    'court_proceedings': "⚖️ **Court Proceedings**: For matters involving active court proceedings, you must consult with a licensed attorney in your jurisdiction. This information cannot substitute for professional legal representation.",
    
    'criminal_matters': "🔒 **Criminal Law**: Criminal law matters require immediate professional legal assistance. Contact a criminal defense attorney or public defender if you cannot afford representation."
}

# User education messages
USER_EDUCATION_MESSAGES = {
    'legal_advice_vs_information': """
**Understanding Legal Information vs. Legal Advice:**
- **Legal Information**: General explanations of laws, procedures, and legal concepts
- **Legal Advice**: Specific guidance about what you should do in your particular situation
- This AI provides legal information only, not legal advice
""",
    
    'when_to_consult_attorney': """
**When to Consult an Attorney:**
- You're facing criminal charges
- You're involved in a lawsuit
- You need to file legal documents
- You're dealing with significant financial or property matters
- The situation involves complex legal issues
""",
    
    'finding_legal_help': """
**Finding Legal Help:**
- Local bar association referral services
- Legal aid organizations for low-income individuals
- Pro bono programs
- Online attorney directories
- Law school clinics
"""
}

# Default safety configuration
DEFAULT_SAFETY_CONFIG = SafetyConfig(
    safety_mode=SafetyMode.STANDARD,
    max_message_length=500,
    enable_profanity_filter=True,
    enable_intent_classification=True,
    block_harmful_content=True,
    require_source_attribution=True,
    add_legal_disclaimers=True,
    log_all_interactions=True,
    enable_guardrails=True
)
