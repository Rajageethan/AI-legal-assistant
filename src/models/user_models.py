from pydantic import BaseModel, Field
try:
    from pydantic import EmailStr
except ImportError:
    # Fallback if email-validator is not installed
    EmailStr = str
from typing import Dict, List, Optional, Any
from datetime import datetime

class UserProfile(BaseModel):
    """User profile model"""
    uid: str = Field(..., description="Firebase user ID")
    email: EmailStr = Field(..., description="User email")
    name: Optional[str] = Field(None, description="User display name")
    email_verified: bool = Field(False, description="Email verification status")
    created_at: Optional[datetime] = Field(None, description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    total_chats: int = Field(0, description="Total number of chats")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

class UserAuth(BaseModel):
    """User authentication model"""
    id_token: str = Field(..., description="Firebase ID token")

class ChatContext(BaseModel):
    """Chat context model for Firestore storage"""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    user_id: str = Field(..., description="User ID")
    title: str = Field("Untitled Chat", description="Chat title")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Chat messages")
    context_memory: Dict[str, Any] = Field(default_factory=dict, description="RAG context and memory")
    created_at: Optional[datetime] = Field(None, description="Chat creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    message_count: int = Field(0, description="Number of messages in conversation")
    last_message: str = Field("", description="Preview of last message")

class ChatHistoryItem(BaseModel):
    """Chat history item for listing user's conversations"""
    conversation_id: str = Field(..., description="Conversation ID")
    title: str = Field(..., description="Chat title")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    message_count: int = Field(0, description="Number of messages")
    last_message: str = Field("", description="Preview of last message")

class AuthResponse(BaseModel):
    """Authentication response model"""
    success: bool = Field(..., description="Authentication success status")
    user: Optional[UserProfile] = Field(None, description="User profile data")
    message: str = Field(..., description="Response message")

class ChatContextResponse(BaseModel):
    """Chat context response model"""
    success: bool = Field(..., description="Operation success status")
    context: Optional[ChatContext] = Field(None, description="Chat context data")
    message: str = Field(..., description="Response message")

class ChatHistoryResponse(BaseModel):
    """Chat history response model"""
    success: bool = Field(..., description="Operation success status")
    history: List[ChatHistoryItem] = Field(default_factory=list, description="Chat history")
    total_count: int = Field(0, description="Total number of conversations")
    message: str = Field(..., description="Response message")
