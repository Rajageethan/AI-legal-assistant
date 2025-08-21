from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(
        ..., 
        description="The user's legal question",
        min_length=1,
        max_length=500,
        example="What are the key elements of a valid contract?"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for context continuity",
        example="abc123def"
    )
    optimize_for_speed: bool = Field(
        True,
        description="Whether to optimize response for speed over detail"
    )

    class Config:
        schema_extra = {
            "example": {
                "message": "What are the key elements of a valid contract under US law?",
                "conversation_id": "conv_123",
                "optimize_for_speed": True
            }
        }

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="AI-generated response to the legal question")
    sources: List[str] = Field(
        default=[],
        description="List of source documents that informed the response"
    )
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    usage_info: Dict[str, Any] = Field(
        default={},
        description="Usage statistics and metadata"
    )
    processing_time: float = Field(..., description="Time taken to process the request (seconds)")
    model_used: str = Field(
        default="llama3-8b-8192",
        description="The AI model used to generate the response"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "response": "A valid contract requires four key elements: offer, acceptance, consideration, and legal capacity...",
                "sources": ["Contract Law Basics (Page 1)", "Legal Fundamentals (Page 15)"],
                "conversation_id": "conv_123",
                "usage_info": {
                    "requests_made": 5,
                    "session_duration": "0:15:30"
                },
                "processing_time": 2.3,
                "model_used": "llama3-8b-8192"
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Overall system health status")
    timestamp: str = Field(..., description="Timestamp of health check")
    version: str = Field(..., description="API version")
    system_info: Dict[str, Any] = Field(..., description="System resource information")
    usage_stats: Dict[str, Any] = Field(..., description="Usage statistics")
    groq_status: Dict[str, Any] = Field(..., description="Groq-specific status information")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-12-07T10:30:00Z",
                "version": "0.3.0",
                "system_info": {
                    "cpu_usage": 15.2,
                    "memory_usage": 45.7,
                    "active_conversations": 3
                },
                "usage_stats": {
                    "total_requests_today": 127,
                    "avg_requests_per_hour": 8.2
                },
                "groq_status": {
                    "api_key_configured": True,
                    "primary_model": "llama3-8b-8192",
                    "rate_limiting_active": True
                }
            }
        }

class UsageStatsResponse(BaseModel):
    """Response model for usage statistics endpoint"""
    current_session: Dict[str, Any] = Field(..., description="Current session statistics")
    rate_limiting: Dict[str, Any] = Field(..., description="Rate limiting information")
    recommendations: List[str] = Field(..., description="Usage optimization recommendations")
    free_tier_status: Dict[str, Any] = Field(..., description="Free tier usage status")
    
    class Config:
        schema_extra = {
            "example": {
                "current_session": {
                    "total_requests": 45,
                    "active_conversations": 2,
                    "session_duration": "1:23:45"
                },
                "rate_limiting": {
                    "requests_per_minute": 12,
                    "daily_requests": 245,
                    "max_daily_requests": 14400
                },
                "recommendations": [
                    "Usage looks optimal for free tier!",
                    "Consider batching similar queries for efficiency"
                ],
                "free_tier_status": {
                    "daily_usage_percentage": 1.7,
                    "requests_remaining": 14155,
                    "rate_limit_status": "healthy"
                }
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    path: str = Field(..., description="Request path where error occurred")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retrying (for rate limits)")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "RateLimitError",
                "message": "Rate limit reached. Please wait before trying again.",
                "timestamp": "2024-12-07T10:30:00Z",
                "path": "/api/chat",
                "retry_after": 60
            }
        }

class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint"""
    primary_model: str = Field(..., description="Primary AI model being used")
    fallback_models: List[str] = Field(..., description="Available fallback models")
    provider: str = Field(..., description="AI service provider")
    optimization_level: str = Field(..., description="Current optimization configuration")
    features: Dict[str, bool] = Field(..., description="Available features")
    limits: Dict[str, Any] = Field(..., description="Current usage limits")
    performance: Dict[str, str] = Field(..., description="Performance characteristics")
    
    class Config:
        schema_extra = {
            "example": {
                "primary_model": "llama3-8b-8192",
                "fallback_models": ["llama-3.1-8b-instant", "gemma-7b-it"],
                "provider": "Groq",
                "optimization_level": "free_tier_maximum",
                "features": {
                    "ultra_fast_inference": True,
                    "smart_rate_limiting": True,
                    "token_optimization": True
                },
                "limits": {
                    "max_tokens_per_request": 800,
                    "max_message_length": 500
                },
                "performance": {
                    "typical_response_time": "1-3 seconds",
                    "groq_lpu_advantage": "10x faster than traditional GPU inference"
                }
            }
        }

class ConversationDeleteResponse(BaseModel):
    """Response model for conversation deletion"""
    message: str = Field(..., description="Confirmation message")
    conversation_id: str = Field(..., description="ID of deleted conversation")
    timestamp: str = Field(..., description="Deletion timestamp")

class OptimizationResponse(BaseModel):
    """Response model for system optimization endpoint"""
    status: str = Field(..., description="Optimization status")
    conversations_cleaned: int = Field(..., description="Number of conversations cleaned up")
    active_conversations: int = Field(..., description="Remaining active conversations")
    memory_optimization: str = Field(..., description="Memory optimization status")
    timestamp: str = Field(..., description="Optimization timestamp")
    error: Optional[str] = Field(None, description="Error message if optimization failed")

# Custom validation functions
def validate_conversation_id(conversation_id: str) -> str:
    """Validate conversation ID format"""
    if not conversation_id or len(conversation_id.strip()) == 0:
        raise ValueError("Conversation ID cannot be empty")
    if len(conversation_id) > 50:
        raise ValueError("Conversation ID too long (max 50 characters)")
    return conversation_id.strip()

def validate_message_content(message: str) -> str:
    """Validate message content for token efficiency"""
    if not message or len(message.strip()) == 0:
        raise ValueError("Message cannot be empty")
    if len(message.strip()) > 500:
        raise ValueError("Message too long for free tier optimization (max 500 characters)")
    return message.strip()