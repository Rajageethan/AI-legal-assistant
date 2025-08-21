from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
import uuid

# Add the parent directory to the Python path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.services.llm_client import GroqLlama32Client  # Import our optimized client
    from src.services.firebase_service import firebase_service
    from src.services.safety_service import safety_service, SafetyLevel, IntentCategory
    from src.services.content_filter import content_filter, guardrails_integration
    from src.models.user_models import (
        UserAuth, UserProfile, ChatContext, ChatHistoryItem,
        AuthResponse, ChatContextResponse, ChatHistoryResponse
    )
    # All imports successful
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure all required modules are in the correct location")
    import traceback
    traceback.print_exc()
    # Continue without raising to see which specific import fails

router = APIRouter()

# Create FastAPI app instance for direct uvicorn usage
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Legal Assistant API",
    description="Legal assistant powered by Groq Llama 3.2",
    version="0.3.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Router will be included at the end of the file after all routes are defined

# Global storage for client instances (memory-based for free tier efficiency)
groq_clients: Dict[str, GroqLlama32Client] = {}
client_last_used: Dict[str, datetime] = {}

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's legal question", max_length=500)
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    optimize_for_speed: bool = Field(True, description="Whether to optimize for speed over detail")
    save_context: bool = Field(True, description="Whether to save chat context to Firebase")
    safety_level: str = Field("standard", description="Safety validation level: basic, standard, strict")

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    conversation_id: str
    usage_info: Dict[str, Any] = {}
    processing_time: float
    model_used: str = "llama3-8b-8192"
    safety_info: Dict[str, Any] = {}
    content_warnings: List[str] = []

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    system_info: Dict[str, Any]
    usage_stats: Dict[str, Any]
    groq_status: Dict[str, Any]

class UsageStatsResponse(BaseModel):
    current_session: Dict[str, Any]
    rate_limiting: Dict[str, Any]
    recommendations: List[str]
    free_tier_status: Dict[str, Any]

async def get_or_create_client(conversation_id: str) -> GroqLlama32Client:
    """Get existing client or create new one with data initialization"""
    
    if conversation_id in groq_clients:
        # Update last used time
        client_last_used[conversation_id] = datetime.now()
        return groq_clients[conversation_id]
    
    # Create new client
    try:
        print(f"🔄 Creating new Groq client for conversation {conversation_id}")
        client = GroqLlama32Client()
        
        # Initialize with legal data if available
        await initialize_client_data(client)
        
        # Store client and track usage
        groq_clients[conversation_id] = client
        client_last_used[conversation_id] = datetime.now()
        
        print(f"Client created and initialized for {conversation_id}")
        return client
        
    except Exception as e:
        print(f"❌ Error creating client: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize AI client: {str(e)}")

async def initialize_client_data(client: GroqLlama32Client):
    """Initialize client with legal document data"""
    try:
        # Look for data files in multiple locations
        data_locations = [
            os.path.join(os.path.dirname(__file__), '..', 'data'),
            os.path.join(os.getcwd(), 'src', 'data'),
            os.path.join(os.getcwd(), 'data')
        ]
        
        # Load both RAG_data.jsonl and faq.jsonl
        rag_file = None
        faq_file = None
        
        for data_dir in data_locations:
            rag_path = os.path.join(data_dir, 'RAG_data.jsonl')
            faq_path = os.path.join(data_dir, 'faq.jsonl')
            
            if os.path.exists(rag_path):
                rag_file = rag_path
            if os.path.exists(faq_path):
                faq_file = faq_path
        
        if not rag_file and not faq_file:
            print("⚠️ Warning: No data files found. Client will work with general knowledge only.")
            return
        
        print(f"📁 Loading data from:")
        if rag_file:
            print(f"   - RAG data: {rag_file}")
        if faq_file:
            print(f"   - FAQ data: {faq_file}")
        
        # Load and process data efficiently for free tier
        texts = []
        metadatas = []
        doc_count = 0
        
        # Process RAG_data.jsonl
        if rag_file:
            with open(rag_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if doc_count >= 150:  # Reserve space for FAQ data
                        break
                        
                    try:
                        doc = json.loads(line.strip())
                        
                        # Extract text and metadata efficiently
                        text = doc.get('content', '').strip()
                        if text and len(text) > 50:  # Only keep substantial content
                            # Truncate very long texts to save tokens
                            if len(text.split()) > 300:
                                text = ' '.join(text.split()[:300]) + "..."
                            
                            texts.append(text)
                            metadatas.append({
                                'title': doc.get('title', f'Legal Document {doc_count + 1}')[:50],
                                'page': doc.get('page', 1),
                                'source': doc.get('source', 'legal_document'),
                                'type': 'legal_text'
                            })
                            doc_count += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping malformed JSON line in RAG data: {e}")
                        continue
        
        # Process faq.jsonl
        if faq_file:
            with open(faq_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if doc_count >= 300:  # Total limit for free tier
                        break
                        
                    try:
                        doc = json.loads(line.strip())
                        
                        # Combine question and answer for better retrieval
                        question = doc.get('question', '').strip()
                        answer = doc.get('answer', '').strip()
                        
                        if question and answer:
                            # Create comprehensive text for embedding
                            text = f"Q: {question}\nA: {answer}"
                            
                            texts.append(text)
                            metadatas.append({
                                'title': question[:50],
                                'question': question,
                                'answer': answer,
                                'source': doc.get('source', 'FAQ'),
                                'type': 'faq',
                                'id': doc.get('id', f'faq-{doc_count}')
                            })
                            doc_count += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping malformed JSON line in FAQ data: {e}")
                        continue
        
        if texts:
            print(f"📊 Processing {len(texts)} documents for vector database...")
            legal_docs = sum(1 for m in metadatas if m.get('type') == 'legal_text')
            faq_docs = sum(1 for m in metadatas if m.get('type') == 'faq')
            print(f"   - Legal documents: {legal_docs}")
            print(f"   - FAQ entries: {faq_docs}")
            
            client.initialize_db(texts, metadatas)
            print(f"✅ Database initialized with {len(texts)} total documents")
        else:
            print("⚠️ No valid documents found in data files")
            
    except Exception as e:
        print(f"⚠️ Error initializing data: {e}")
        print("Client will continue with general legal knowledge only")

async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Get current user from Firebase token"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    try:
        id_token = authorization.split("Bearer ")[1]
        user_info = await firebase_service.verify_user_token(id_token)
        return user_info
    except Exception as e:
        print(f"❌ Error verifying user token: {e}")
        return None

def cleanup_inactive_clients():
    """Clean up inactive clients to save memory (free tier optimization)"""
    cutoff_time = datetime.now() - timedelta(hours=1)  # 1 hour timeout
    inactive_clients = []
    
    for conv_id, last_used in client_last_used.items():
        if last_used < cutoff_time:
            inactive_clients.append(conv_id)
    
    for conv_id in inactive_clients:
        if conv_id in groq_clients:
            del groq_clients[conv_id]
        if conv_id in client_last_used:
            del client_last_used[conv_id]
        print(f"🧹 Cleaned up inactive client: {conv_id}")
    
    if inactive_clients:
        print(f"🧹 Cleaned up {len(inactive_clients)} inactive conversations")

async def save_enhanced_chat_context(
    user_id: str,
    conversation_id: str,
    user_message: str,
    bot_response: str,
    sources: List[str],
    safety_info: Dict[str, Any]
):
    """Enhanced context saving with safety metadata"""
    try:
        # Get existing context
        existing_context = await firebase_service.get_chat_context(user_id, conversation_id)
        
        if existing_context:
            messages = existing_context.get('messages', [])
            context_memory = existing_context.get('context_memory', {})
            title = existing_context.get('title', 'Legal Chat')
        else:
            messages = []
            context_memory = {}
            title = user_message[:50] + "..." if len(user_message) > 50 else user_message
        
        # Add new message with safety metadata
        messages.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'sources': sources,
            'safety_info': safety_info
        })
        
        # Update context memory
        context_memory.update({
            'last_sources': sources,
            'message_count': len(messages),
            'last_updated': datetime.now().isoformat(),
            'safety_summary': {
                'total_warnings': sum(1 for msg in messages if msg.get('safety_info', {}).get('input_validation', {}).get('safety_level') == 'warning'),
                'content_filtered': sum(1 for msg in messages if msg.get('safety_info', {}).get('content_filtering', {}).get('input_filtered', False))
            }
        })
        
        # Prepare enhanced context data
        context_data = {
            'title': title,
            'messages': messages[-10:],  # Keep last 10 messages
            'context_memory': context_memory,
            'message_count': len(messages),
            'last_message': user_message[:200],
            'safety_metadata': {
                'last_safety_check': datetime.now().isoformat(),
                'safety_level': safety_info.get('input_validation', {}).get('safety_level', 'safe')
            }
        }
        
        # Save to Firebase
        await firebase_service.save_chat_context(user_id, conversation_id, context_data)
        await firebase_service.update_chat_metadata(user_id, conversation_id, title, user_message)
        
        print(f"✅ Saved enhanced chat context with safety metadata: {conversation_id}")
        
    except Exception as e:
        print(f"❌ Error saving enhanced chat context: {e}")

async def save_chat_context_to_firebase(
    user_id: str, 
    conversation_id: str, 
    user_message: str, 
    bot_response: str, 
    sources: List[str]
):
    """Legacy function - redirects to enhanced version"""
    await save_enhanced_chat_context(
        user_id, conversation_id, user_message, bot_response, sources, {}
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Enhanced chat endpoint with comprehensive safety validation"""
    start_time = asyncio.get_event_loop().time()
    conversation_id = request.conversation_id or str(uuid.uuid4())[:8]
    
    try:
        # Step 1: Validate user input for safety
        print(f"🔍 Validating user input for safety...")
        input_validation = await safety_service.validate_user_input(
            request.message, 
            user_context={'user_id': current_user.get('uid') if current_user else None}
        )
        
        # Block unsafe content
        if not input_validation['is_safe']:
            await safety_service.log_safety_event("input_blocked", {
                'reason': input_validation['reason'],
                'user_id': current_user.get('uid') if current_user else 'anonymous',
                'message_preview': request.message[:50]
            })
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Content Safety Violation",
                    "message": input_validation.get('suggested_response', 
                        "Your request cannot be processed due to safety concerns."),
                    "safety_info": input_validation
                }
            )
        
        # Step 2: Apply content filtering
        print(f"🛡️ Applying content filters...")
        content_filter_result = content_filter.filter_user_input(
            request.message,
            user_context={'user_id': current_user.get('uid') if current_user else None}
        )
        
        if content_filter_result['is_filtered']:
            await safety_service.log_safety_event("content_filtered", {
                'filter_reason': content_filter_result['filter_reason'],
                'severity': content_filter_result['severity'],
                'categories': content_filter_result['categories']
            })
            processed_message = content_filter_result['filtered_message']
        else:
            processed_message = request.message.strip()
        
        # Optimize message for free tier (truncate if too long)
        if len(processed_message.split()) > 100:
            processed_message = ' '.join(processed_message.split()[:100]) + "..."
            print(f"📝 Message truncated for token efficiency")
        
        # Step 3: Schedule cleanup and get client
        background_tasks.add_task(cleanup_inactive_clients)
        
        # Load existing context from Firebase if user is authenticated
        existing_context = None
        if current_user and firebase_service.is_initialized():
            existing_context = await firebase_service.get_chat_context(
                current_user['uid'], conversation_id
            )
        
        client = await get_or_create_client(conversation_id)
        
        # Restore context if available
        if existing_context and existing_context.get('context_memory'):
            print(f"🔄 Restoring context for conversation {conversation_id}")
        
        # Step 4: Get AI response
        print(f"🤖 Processing query: {processed_message[:50]}...")
        response_text, sources = await client.get_response(processed_message)
        
        # Step 5: Validate AI response
        print(f"🔍 Validating AI response for safety...")
        response_validation = await safety_service.validate_bot_response(
            response_text, processed_message, sources
        )
        
        # Step 6: Apply response filtering
        response_filter_result = content_filter.filter_bot_response(response_text, sources)
        
        # Step 7: Enhance response if needed
        if response_validation['needs_modification'] or response_filter_result['is_filtered']:
            print(f"⚠️ Enhancing response for safety...")
            response_text = safety_service.generate_safety_enhanced_response(
                response_text, response_validation, processed_message
            )
            
            if response_filter_result['is_filtered']:
                response_text = response_filter_result['filtered_response']
        
        # Step 8: Apply Guardrails validation if available
        if request.safety_level in ['standard', 'strict']:
            guardrails_result = await guardrails_integration.validate_with_guardrails(
                response_text, "legal_response"
            )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Step 9: Prepare safety information
        safety_info = {
            'input_validation': {
                'safety_level': input_validation['safety_level'].value,
                'intent': input_validation['intent'].value,
                'requires_disclaimer': input_validation['requires_disclaimer']
            },
            'response_validation': {
                'quality_score': response_validation['quality_score'],
                'needs_modification': response_validation['needs_modification'],
                'issues': response_validation['issues']
            },
            'content_filtering': {
                'input_filtered': content_filter_result['is_filtered'],
                'response_filtered': response_filter_result['is_filtered']
            }
        }
        
        # Collect content warnings
        content_warnings = []
        if input_validation['safety_level'] == SafetyLevel.WARNING:
            content_warnings.append("This request may be seeking legal advice")
        if input_validation.get('content_flags'):
            content_warnings.extend([f"Flagged: {flag}" for flag in input_validation['content_flags']])
        
        # Get usage stats for monitoring
        usage_stats = client.get_usage_stats() if hasattr(client, 'get_usage_stats') else {}
        
        # Step 10: Save context with safety metadata
        if current_user and request.save_context and firebase_service.is_initialized():
            background_tasks.add_task(
                save_enhanced_chat_context,
                current_user['uid'],
                conversation_id,
                processed_message,
                response_text,
                sources,
                safety_info
            )
        
        # Step 11: Log successful interaction
        await safety_service.log_safety_event("safe_interaction", {
            'user_id': current_user.get('uid') if current_user else 'anonymous',
            'intent': input_validation['intent'].value,
            'safety_level': input_validation['safety_level'].value,
            'processing_time': processing_time
        })
        
        # Log performance metrics
        print(f"⚡ Response generated in {processing_time:.2f}s")
        if processing_time < 2:
            print("Groq speed advantage evident!")
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            conversation_id=conversation_id,
            usage_info=usage_stats,
            processing_time=round(processing_time, 3),
            model_used="llama3-8b-8192",
            safety_info=safety_info,
            content_warnings=content_warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = asyncio.get_event_loop().time() - start_time
        error_msg = str(e)
        
        print(f"❌ Error in chat endpoint: {error_msg}")
        traceback.print_exc()
        
        # Log safety event for errors
        await safety_service.log_safety_event("endpoint_error", {
            'error': error_msg,
            'user_id': current_user.get('uid') if current_user else 'anonymous',
            'processing_time': processing_time
        })
        
        # Enhanced error handling for Groq-specific issues
        if "rate" in error_msg.lower() or "limit" in error_msg.lower():
            raise HTTPException(
                status_code=429, 
                detail="Rate limit reached. Please wait a moment before trying again. The free tier has daily limits."
            )
        elif "quota" in error_msg.lower() or "credit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Daily quota exceeded. Please try again tomorrow or consider upgrading for higher limits."
            )
        elif "token" in error_msg.lower() and "invalid" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Invalid API token. Please check your Groq API key configuration."
            )
        else:
            # Return safe fallback response
            fallback_response = "I apologize, but I'm experiencing technical difficulties. For immediate legal assistance, please consult with a qualified attorney in your jurisdiction."
            
            return ChatResponse(
                response=fallback_response,
                sources=[],
                conversation_id=conversation_id,
                usage_info={"error": "fallback_response_used"},
                processing_time=round(processing_time, 3),
                model_used="fallback",
                safety_info={"error": "system_error", "fallback_used": True},
                content_warnings=["System error - fallback response provided"]
            )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with Groq-specific monitoring"""
    try:
        import psutil
        import platform
        from datetime import datetime
        
        # System information
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            "active_conversations": len(groq_clients),
        }
        
        # Aggregate usage statistics
        total_requests = 0
        avg_requests_per_hour = 0
        
        for client in groq_clients.values():
            if hasattr(client, 'request_count'):
                total_requests += client.request_count
            if hasattr(client, 'get_usage_stats'):
                stats = client.get_usage_stats()
                avg_requests_per_hour += stats.get('avg_requests_per_hour', 0)
        
        usage_stats = {
            "total_requests_today": total_requests,
            "active_conversations": len(groq_clients),
            "avg_requests_per_hour": round(avg_requests_per_hour / max(len(groq_clients), 1), 2),
            "clients_created": len(client_last_used),
        }
        
        # Groq-specific status
        groq_status = {
            "api_key_configured": bool(os.getenv("GROQ_API_KEY")),
            "api_key_format_valid": os.getenv("GROQ_API_KEY", "").startswith("gsk_") if os.getenv("GROQ_API_KEY") else False,
            "primary_model": "llama3-8b-8192",
            "optimization_level": "free_tier_max",
            "rate_limiting_active": True,
            "local_embeddings": True,
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="0.3.0",
            system_info=system_info,
            usage_stats=usage_stats,
            groq_status=groq_status
        )
        
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
            version="0.3.0",
            system_info={"error": str(e)},
            usage_stats={},
            groq_status={"status": "error", "detail": str(e)}
        )

@router.get("/usage-stats", response_model=UsageStatsResponse)
async def get_usage_stats():
    """Detailed usage statistics for free tier monitoring"""
    try:
        # Aggregate current session stats
        total_requests = 0
        session_duration = timedelta(0)
        oldest_client = datetime.now()
        
        rate_limit_info = {
            "requests_per_minute": 0,
            "daily_requests": 0,
            "estimated_tokens_used": 0,
        }
        
        for conv_id, client in groq_clients.items():
            if hasattr(client, 'get_usage_stats'):
                stats = client.get_usage_stats()
                total_requests += stats.get('requests_made', 0)
                
                if hasattr(client, 'session_start'):
                    client_duration = datetime.now() - client.session_start
                    if client_duration > session_duration:
                        session_duration = client_duration
                        oldest_client = client.session_start
                
                # Get rate limiting info from first client
                if conv_id == list(groq_clients.keys())[0]:
                    if hasattr(client.llm, 'rate_limiter'):
                        limiter = client.llm.rate_limiter
                        rate_limit_info = {
                            "requests_per_minute": len(limiter.request_times),
                            "daily_requests": limiter.daily_requests,
                            "max_daily_requests": limiter.max_requests_per_day,
                            "max_requests_per_minute": limiter.max_requests_per_minute,
                        }
        
        current_session = {
            "total_requests": total_requests,
            "active_conversations": len(groq_clients),
            "session_duration": str(session_duration),
            "avg_requests_per_conversation": round(total_requests / max(len(groq_clients), 1), 2),
            "oldest_conversation": oldest_client.isoformat() if oldest_client != datetime.now() else None,
        }
        
        # Generate recommendations based on usage
        recommendations = []
        
        if rate_limit_info.get("daily_requests", 0) > 10000:
            recommendations.append("⚠️ Approaching daily limit. Consider optimizing query frequency.")
        
        if rate_limit_info.get("requests_per_minute", 0) > 20:
            recommendations.append("🐌 High request frequency detected. Consider batching queries.")
        
        if total_requests > 50:
            recommendations.append("📊 Heavy usage session. Monitor free tier limits closely.")
        
        if len(groq_clients) > 5:
            recommendations.append("🧹 Multiple conversations active. Consider consolidating for efficiency.")
        
        if not recommendations:
            recommendations.append("Usage looks optimal for free tier!")
        
        # Free tier status assessment
        daily_usage_percent = (rate_limit_info.get("daily_requests", 0) / 
                             rate_limit_info.get("max_daily_requests", 14400)) * 100
        
        free_tier_status = {
            "daily_usage_percentage": round(daily_usage_percent, 1),
            "requests_remaining": rate_limit_info.get("max_daily_requests", 14400) - rate_limit_info.get("daily_requests", 0),
            "rate_limit_status": "healthy" if rate_limit_info.get("requests_per_minute", 0) < 25 else "approaching_limit",
            "optimization_active": True,
            "estimated_cost_saved": f"~${(total_requests * 0.0001):.4f}",  # Rough estimate
        }
        
        return UsageStatsResponse(
            current_session=current_session,
            rate_limiting=rate_limit_info,
            recommendations=recommendations,
            free_tier_status=free_tier_status
        )
        
    except Exception as e:
        print(f"❌ Error getting usage stats: {e}")
        return UsageStatsResponse(
            current_session={"error": str(e)},
            rate_limiting={"error": "Could not retrieve rate limiting info"},
            recommendations=["⚠️ Error retrieving usage statistics"],
            free_tier_status={"status": "unknown", "error": str(e)}
        )

@router.get("/model-info")
async def get_model_info():
    """Get information about the current Groq model configuration"""
    return {
        "primary_model": "llama3-8b-8192",
        "fallback_models": ["llama-3.1-8b-instant", "gemma-7b-it"],
        "provider": "Groq",
        "optimization_level": "free_tier_maximum",
        "features": {
            "ultra_fast_inference": True,
            "local_embeddings": True,
            "smart_rate_limiting": True,
            "token_optimization": True,
            "automatic_fallbacks": True,
            "conversation_memory": True,
        },
        "limits": {
            "max_tokens_per_request": 800,
            "max_message_length": 500,
            "max_documents_processed": 200,
            "conversation_timeout": "1 hour",
        },
        "performance": {
            "typical_response_time": "1-3 seconds",
            "groq_lpu_advantage": "10x faster than traditional GPU inference",
        }
    }

@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation to free up memory"""
    if conversation_id in groq_clients:
        del groq_clients[conversation_id]
        if conversation_id in client_last_used:
            del client_last_used[conversation_id]
        return {"message": f"Conversation {conversation_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@router.post("/optimize")
async def optimize_system():
    """Manually trigger system optimization for free tier efficiency"""
    try:
        initial_clients = len(groq_clients)
        
        # Clean up inactive clients
        cleanup_inactive_clients()
        
        # Force garbage collection if available
        try:
            import gc
            gc.collect()
        except:
            pass
        
        final_clients = len(groq_clients)
        cleaned_up = initial_clients - final_clients
        
        return {
            "status": "optimization_complete",
            "conversations_cleaned": cleaned_up,
            "active_conversations": final_clients,
            "memory_optimization": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "optimization_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Background task to periodically clean up inactive clients
import asyncio
from fastapi import BackgroundTasks

async def periodic_cleanup():
    """Periodic cleanup task to maintain optimal performance"""
    while True:
        try:
            cleanup_inactive_clients()
            await asyncio.sleep(1800)  # Run every 30 minutes
        except Exception as e:
            print(f"❌ Error in periodic cleanup: {e}")
            await asyncio.sleep(1800)  # Continue trying

# Authentication and user management endpoints

@router.post("/auth/login", response_model=AuthResponse)
async def login_user(auth_request: UserAuth):
    """Authenticate user with Firebase ID token"""
    try:
        if not firebase_service.is_initialized():
            # Return a more helpful error message for setup
            raise HTTPException(
                status_code=503, 
                detail="Firebase not configured. Please set up Firebase credentials in .env file"
            )
        
        # Verify the Firebase ID token
        user_info = await firebase_service.verify_user_token(auth_request.id_token)
        
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Create or update user profile
        await firebase_service.create_user_profile(user_info['uid'], user_info)
        
        # Get full user profile
        user_profile = await firebase_service.get_user_profile(user_info['uid'])
        
        # Add uid to user profile data for UserProfile model
        if user_profile:
            user_profile['uid'] = user_info['uid']
        
        return AuthResponse(
            success=True,
            user=UserProfile(**user_profile) if user_profile else None,
            message="Authentication successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@router.post("/auth/signup", response_model=AuthResponse)
async def signup_user(auth_request: UserAuth):
    """Sign up user with Firebase ID token"""
    try:
        if not firebase_service.is_initialized():
            # Return a more helpful error message for setup
            raise HTTPException(
                status_code=503, 
                detail="Firebase not configured. Please set up Firebase credentials in .env file"
            )
        
        # Verify the Firebase ID token (same as login since Firebase handles signup client-side)
        user_info = await firebase_service.verify_user_token(auth_request.id_token)
        
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        # Create user profile (will create new or update existing)
        await firebase_service.create_user_profile(user_info['uid'], user_info)
        
        # Get full user profile
        user_profile = await firebase_service.get_user_profile(user_info['uid'])
        
        return AuthResponse(
            success=True,
            user=UserProfile(**user_profile) if user_profile else None,
            message="Account created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Signup error: {e}")
        raise HTTPException(status_code=500, detail="Account creation failed")

@router.get("/auth/profile", response_model=AuthResponse)
async def get_user_profile_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user profile"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        user_profile = await firebase_service.get_user_profile(current_user['uid'])
        
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return AuthResponse(
            success=True,
            user=UserProfile(**user_profile),
            message="Profile retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Profile retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")

@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get user's chat history"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        if not firebase_service.is_initialized():
            raise HTTPException(status_code=503, detail="Firebase service not available")
        
        chat_history = await firebase_service.get_user_chat_history(current_user['uid'], limit)
        
        history_items = [ChatHistoryItem(**item) for item in chat_history]
        
        return ChatHistoryResponse(
            success=True,
            history=history_items,
            total_count=len(history_items),
            message="Chat history retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@router.get("/chat/context/{conversation_id}", response_model=ChatContextResponse)
async def get_chat_context_endpoint(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get specific chat context"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        if not firebase_service.is_initialized():
            raise HTTPException(status_code=503, detail="Firebase service not available")
        
        context = await firebase_service.get_chat_context(current_user['uid'], conversation_id)
        
        if not context:
            raise HTTPException(status_code=404, detail="Chat context not found")
        
        return ChatContextResponse(
            success=True,
            context=ChatContext(**context),
            message="Chat context retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat context error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat context")

@router.delete("/chat/context/{conversation_id}")
async def delete_chat_context_endpoint(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete specific chat context"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        if not firebase_service.is_initialized():
            raise HTTPException(status_code=503, detail="Firebase service not available")
        
        success = await firebase_service.delete_chat_context(current_user['uid'], conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Chat context not found")
        
        # Also clean up local client if exists
        if conversation_id in groq_clients:
            del groq_clients[conversation_id]
        if conversation_id in client_last_used:
            del client_last_used[conversation_id]
        
        return {"message": f"Chat context {conversation_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Delete chat context error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat context")

# Add new safety endpoints
@router.post("/validate-input")
async def validate_input_endpoint(
    message: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Endpoint to validate user input before processing"""
    try:
        validation_result = await safety_service.validate_user_input(
            message,
            user_context={'user_id': current_user.get('uid') if current_user else None}
        )
        
        return {
            "is_safe": validation_result['is_safe'],
            "safety_level": validation_result['safety_level'].value,
            "intent_category": validation_result['intent'].value,
            "requires_disclaimer": validation_result['requires_disclaimer'],
            "content_flags": validation_result.get('content_flags', []),
            "recommendations": validation_result.get('recommendations', [])
        }
        
    except Exception as e:
        print(f"❌ Error in input validation: {e}")
        raise HTTPException(status_code=500, detail="Validation service error")

@router.get("/safety-stats")
async def get_safety_stats(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Get safety statistics and monitoring information"""
    try:
        from src.services.monitoring_service import safety_monitor
        return safety_monitor.get_safety_dashboard()
        
    except Exception as e:
        print(f"❌ Error getting safety stats: {e}")
        return {
            "safety_events_today": 0,
            "blocked_requests": 0,
            "filtered_responses": 0,
            "warning_level_interactions": 0,
            "top_safety_concerns": [],
            "system_health": {
                "safety_service": "operational",
                "content_filter": "operational",
                "guardrails": "available" if guardrails_integration.is_available else "unavailable"
            },
            "error": str(e)
        }

# Include router with /api prefix to match frontend expectations
# This must be at the end after all route definitions
app.include_router(router, prefix="/api")

# Note: This background task would be started in main.py startup event