"""
Safety monitoring and logging service for legal chatbot
Tracks safety events, user behavior patterns, and system health
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum

class EventType(Enum):
    INPUT_BLOCKED = "input_blocked"
    CONTENT_FILTERED = "content_filtered"
    RESPONSE_MODIFIED = "response_modified"
    SAFE_INTERACTION = "safe_interaction"
    BOUNDARY_VIOLATION = "boundary_violation"
    MISINFORMATION_DETECTED = "misinformation_detected"
    SYSTEM_ERROR = "system_error"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SafetyEvent:
    timestamp: datetime
    event_type: EventType
    risk_level: RiskLevel
    user_id: Optional[str]
    details: Dict[str, Any]
    session_id: Optional[str] = None
    ip_address: Optional[str] = None

class SafetyMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.events: List[SafetyEvent] = []
        self.user_patterns: Dict[str, List[SafetyEvent]] = defaultdict(list)
        self.daily_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Risk thresholds
        self.risk_thresholds = {
            'blocked_requests_per_hour': 5,
            'filtered_content_per_hour': 10,
            'boundary_violations_per_day': 3,
            'misinformation_attempts_per_day': 2
        }
        
        # Pattern detection settings
        self.pattern_detection = {
            'min_events_for_pattern': 3,
            'pattern_time_window': timedelta(hours=1),
            'suspicious_user_threshold': 5
        }

    async def log_event(self, event_type: EventType, details: Dict[str, Any], 
                       user_id: Optional[str] = None, risk_level: RiskLevel = RiskLevel.LOW):
        """Log a safety event with automatic risk assessment"""
        
        event = SafetyEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            risk_level=risk_level,
            user_id=user_id,
            details=details
        )
        
        # Store event
        self.events.append(event)
        if user_id:
            self.user_patterns[user_id].append(event)
        
        # Update daily stats
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_stats[today][event_type.value] += 1
        
        # Check for immediate risks
        await self._check_immediate_risks(event)
        
        # Log to system logger
        self.logger.info(f"Safety Event: {event_type.value}", extra={
            'event_type': event_type.value,
            'risk_level': risk_level.value,
            'user_id': user_id,
            'details': details
        })

    async def _check_immediate_risks(self, event: SafetyEvent):
        """Check for immediate risk patterns requiring attention"""
        
        # Check user-specific patterns
        if event.user_id:
            await self._check_user_risk_patterns(event.user_id)
        
        # Check system-wide patterns
        await self._check_system_risk_patterns()
        
        # Check for critical events
        if event.risk_level == RiskLevel.CRITICAL:
            await self._handle_critical_event(event)

    async def _check_user_risk_patterns(self, user_id: str):
        """Check for risky patterns from specific user"""
        user_events = self.user_patterns[user_id]
        recent_events = [e for e in user_events if 
                        datetime.now() - e.timestamp < self.pattern_detection['pattern_time_window']]
        
        # Check for rapid successive violations
        if len(recent_events) >= self.pattern_detection['suspicious_user_threshold']:
            await self._flag_suspicious_user(user_id, recent_events)
        
        # Check for specific violation patterns
        violation_types = Counter(e.event_type for e in recent_events)
        
        if violation_types[EventType.BOUNDARY_VIOLATION] >= 3:
            await self._flag_boundary_violation_pattern(user_id)
        
        if violation_types[EventType.INPUT_BLOCKED] >= 5:
            await self._flag_persistent_inappropriate_requests(user_id)

    async def _check_system_risk_patterns(self):
        """Check for system-wide risk patterns"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        recent_events = [e for e in self.events if e.timestamp > hour_ago]
        event_counts = Counter(e.event_type for e in recent_events)
        
        # Check against thresholds
        if event_counts[EventType.INPUT_BLOCKED] > self.risk_thresholds['blocked_requests_per_hour']:
            await self._alert_high_block_rate()
        
        if event_counts[EventType.CONTENT_FILTERED] > self.risk_thresholds['filtered_content_per_hour']:
            await self._alert_high_filter_rate()

    async def _flag_suspicious_user(self, user_id: str, events: List[SafetyEvent]):
        """Flag user showing suspicious patterns"""
        self.logger.warning(f"Suspicious user pattern detected: {user_id}", extra={
            'user_id': user_id,
            'event_count': len(events),
            'event_types': [e.event_type.value for e in events]
        })

    async def _flag_boundary_violation_pattern(self, user_id: str):
        """Flag repeated boundary violations"""
        self.logger.warning(f"Boundary violation pattern: {user_id}", extra={
            'user_id': user_id,
            'pattern_type': 'boundary_violations'
        })

    async def _flag_persistent_inappropriate_requests(self, user_id: str):
        """Flag persistent inappropriate requests"""
        self.logger.warning(f"Persistent inappropriate requests: {user_id}", extra={
            'user_id': user_id,
            'pattern_type': 'inappropriate_requests'
        })

    async def _alert_high_block_rate(self):
        """Alert for high system block rate"""
        self.logger.error("High system block rate detected", extra={
            'alert_type': 'high_block_rate',
            'threshold_exceeded': self.risk_thresholds['blocked_requests_per_hour']
        })

    async def _alert_high_filter_rate(self):
        """Alert for high content filter rate"""
        self.logger.warning("High content filter rate detected", extra={
            'alert_type': 'high_filter_rate',
            'threshold_exceeded': self.risk_thresholds['filtered_content_per_hour']
        })

    async def _handle_critical_event(self, event: SafetyEvent):
        """Handle critical safety events"""
        self.logger.critical(f"Critical safety event: {event.event_type.value}", extra={
            'event': asdict(event),
            'requires_immediate_attention': True
        })

    def get_safety_dashboard(self) -> Dict[str, Any]:
        """Generate safety monitoring dashboard data"""
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        # Recent events summary
        last_24h = [e for e in self.events if now - e.timestamp < timedelta(days=1)]
        last_hour = [e for e in self.events if now - e.timestamp < timedelta(hours=1)]
        
        # Event type distribution
        event_distribution = Counter(e.event_type.value for e in last_24h)
        
        # Risk level distribution
        risk_distribution = Counter(e.risk_level.value for e in last_24h)
        
        # User activity patterns
        user_activity = Counter(e.user_id for e in last_24h if e.user_id)
        
        # Safety metrics
        total_interactions = len(last_24h)
        safe_interactions = len([e for e in last_24h if e.event_type == EventType.SAFE_INTERACTION])
        safety_rate = (safe_interactions / max(total_interactions, 1)) * 100
        
        return {
            'timestamp': now.isoformat(),
            'summary': {
                'total_events_24h': len(last_24h),
                'total_events_1h': len(last_hour),
                'safety_rate_percent': round(safety_rate, 2),
                'active_users_24h': len(user_activity)
            },
            'event_distribution': dict(event_distribution),
            'risk_distribution': dict(risk_distribution),
            'top_users_by_activity': dict(user_activity.most_common(10)),
            'daily_stats': dict(self.daily_stats[today]),
            'system_health': {
                'within_normal_thresholds': self._check_system_health(),
                'alerts_active': self._get_active_alerts()
            }
        }

    def _check_system_health(self) -> bool:
        """Check if system is operating within normal safety thresholds"""
        now = datetime.now()
        last_hour = [e for e in self.events if now - e.timestamp < timedelta(hours=1)]
        
        event_counts = Counter(e.event_type for e in last_hour)
        
        return (
            event_counts[EventType.INPUT_BLOCKED] <= self.risk_thresholds['blocked_requests_per_hour'] and
            event_counts[EventType.CONTENT_FILTERED] <= self.risk_thresholds['filtered_content_per_hour']
        )

    def _get_active_alerts(self) -> List[str]:
        """Get list of active safety alerts"""
        alerts = []
        
        now = datetime.now()
        last_hour = [e for e in self.events if now - e.timestamp < timedelta(hours=1)]
        event_counts = Counter(e.event_type for e in last_hour)
        
        if event_counts[EventType.INPUT_BLOCKED] > self.risk_thresholds['blocked_requests_per_hour']:
            alerts.append("High block rate")
        
        if event_counts[EventType.CONTENT_FILTERED] > self.risk_thresholds['filtered_content_per_hour']:
            alerts.append("High filter rate")
        
        # Check for critical events in last hour
        critical_events = [e for e in last_hour if e.risk_level == RiskLevel.CRITICAL]
        if critical_events:
            alerts.append(f"{len(critical_events)} critical events")
        
        return alerts

    def get_user_safety_profile(self, user_id: str) -> Dict[str, Any]:
        """Get safety profile for specific user"""
        user_events = self.user_patterns.get(user_id, [])
        
        if not user_events:
            return {'user_id': user_id, 'status': 'no_activity'}
        
        # Recent activity
        now = datetime.now()
        last_24h = [e for e in user_events if now - e.timestamp < timedelta(days=1)]
        last_week = [e for e in user_events if now - e.timestamp < timedelta(days=7)]
        
        # Event analysis
        event_distribution = Counter(e.event_type.value for e in user_events)
        risk_distribution = Counter(e.risk_level.value for e in user_events)
        
        # Risk assessment
        risk_score = self._calculate_user_risk_score(user_events)
        risk_category = self._categorize_user_risk(risk_score)
        
        return {
            'user_id': user_id,
            'risk_category': risk_category,
            'risk_score': risk_score,
            'total_events': len(user_events),
            'events_24h': len(last_24h),
            'events_7d': len(last_week),
            'event_distribution': dict(event_distribution),
            'risk_distribution': dict(risk_distribution),
            'first_seen': min(e.timestamp for e in user_events).isoformat(),
            'last_seen': max(e.timestamp for e in user_events).isoformat(),
            'flags': self._get_user_flags(user_events)
        }

    def _calculate_user_risk_score(self, events: List[SafetyEvent]) -> float:
        """Calculate risk score for user based on event history"""
        if not events:
            return 0.0
        
        # Weight events by type and recency
        score = 0.0
        now = datetime.now()
        
        for event in events:
            # Recency factor (more recent = higher weight)
            days_ago = (now - event.timestamp).days
            recency_factor = max(0.1, 1.0 - (days_ago * 0.1))
            
            # Event type weights
            type_weights = {
                EventType.SAFE_INTERACTION: -0.1,
                EventType.CONTENT_FILTERED: 0.2,
                EventType.INPUT_BLOCKED: 0.5,
                EventType.BOUNDARY_VIOLATION: 1.0,
                EventType.MISINFORMATION_DETECTED: 0.8,
                EventType.SYSTEM_ERROR: 0.1
            }
            
            # Risk level multipliers
            risk_multipliers = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 2.0,
                RiskLevel.HIGH: 3.0,
                RiskLevel.CRITICAL: 5.0
            }
            
            event_score = (
                type_weights.get(event.event_type, 0.3) * 
                risk_multipliers[event.risk_level] * 
                recency_factor
            )
            
            score += event_score
        
        return max(0.0, min(10.0, score))  # Normalize to 0-10 scale

    def _categorize_user_risk(self, risk_score: float) -> str:
        """Categorize user risk based on score"""
        if risk_score < 1.0:
            return "low"
        elif risk_score < 3.0:
            return "medium"
        elif risk_score < 6.0:
            return "high"
        else:
            return "critical"

    def _get_user_flags(self, events: List[SafetyEvent]) -> List[str]:
        """Get behavioral flags for user"""
        flags = []
        
        event_counts = Counter(e.event_type for e in events)
        
        if event_counts[EventType.BOUNDARY_VIOLATION] >= 2:
            flags.append("repeated_boundary_violations")
        
        if event_counts[EventType.INPUT_BLOCKED] >= 5:
            flags.append("persistent_inappropriate_requests")
        
        if event_counts[EventType.MISINFORMATION_DETECTED] >= 2:
            flags.append("misinformation_attempts")
        
        # Check for rapid successive events
        recent_events = [e for e in events if 
                        datetime.now() - e.timestamp < timedelta(hours=1)]
        if len(recent_events) >= 5:
            flags.append("high_frequency_violations")
        
        return flags

# Global monitoring instance
safety_monitor = SafetyMonitor()
