from typing import List, Dict, Tuple, Optional, Any
import os
import requests
import json
import time
import asyncio
from datetime import datetime, timedelta
from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
import chromadb
from dotenv import load_dotenv
import threading
from collections import deque

class GroqRateLimiter:
    """Rate limiter to efficiently manage Groq free tier limits"""
    
    def __init__(self):
        self.request_times = deque()
        self.token_usage = deque()
        self.lock = threading.Lock()
        
        # Groq free tier limits (conservative estimates)
        self.max_requests_per_minute = 30  # Conservative limit
        self.max_tokens_per_minute = 6000   # Conservative limit
        self.max_requests_per_day = 14400   # From rate limits info
        
        # Track daily usage
        self.daily_requests = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if we can make a request without hitting limits"""
        with self.lock:
            now = datetime.now()
            
            # Reset daily counter if needed
            if now >= self.daily_reset_time:
                self.daily_requests = 0
                self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
            # Check daily limits
            if self.daily_requests >= self.max_requests_per_day:
                return False
            
            # Clean old entries (older than 1 minute)
            cutoff_time = now - timedelta(minutes=1)
            while self.request_times and self.request_times[0] < cutoff_time:
                self.request_times.popleft()
            
            while self.token_usage and self.token_usage[0][0] < cutoff_time:
                self.token_usage.popleft()
            
            # Check per-minute limits
            if len(self.request_times) >= self.max_requests_per_minute:
                return False
            
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.max_tokens_per_minute:
                return False
            
            return True
    
    def record_request(self, tokens_used: int = 1000):
        """Record a successful request"""
        with self.lock:
            now = datetime.now()
            self.request_times.append(now)
            self.token_usage.append((now, tokens_used))
            self.daily_requests += 1
    
    def get_wait_time(self) -> int:
        """Get recommended wait time in seconds"""
        with self.lock:
            if not self.request_times:
                return 0
            
            # Time until oldest request expires
            oldest_request = self.request_times[0]
            wait_until = oldest_request + timedelta(minutes=1)
            now = datetime.now()
            
            if wait_until > now:
                return max(1, int((wait_until - now).total_seconds()))
            
            return 1  # Minimum wait time

class GroqLlama32LLM(LLM):
    """Optimized Groq LLM wrapper for Llama 3.2 free tier"""
    
    groq_api_key: str
    model_name: str = "llama3-8b-8192"  # Free tier model
    rate_limiter: Optional[GroqRateLimiter] = None  # Declare rate_limiter as a field
    max_tokens: int = 800  # Default max tokens for responses
    temperature: float = 0.3  # Default temperature for more focused responses
    
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        super().__init__(groq_api_key=api_key, model_name=model_name)
        self.rate_limiter = GroqRateLimiter()
        self._api_url = "https://api.groq.com/openai/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Free tier optimized models in priority order
        self._models = [
            "llama3-8b-8192",      # Primary choice
            "llama-3.1-8b-instant", # Backup
            "gemma-7b-it",         # Fallback
        ]
        
        # Token optimization settings
        self.max_tokens = 800  # Conservative to save tokens
        self.temperature = 0.3  # Lower for more consistent responses
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make optimized API call to Groq"""
        
        # Initialize rate limiter if not already done
        if not hasattr(self, 'rate_limiter') or self.rate_limiter is None:
            self.rate_limiter = GroqRateLimiter()
        
        # Estimate tokens (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3 + self.max_tokens
        
        # Check rate limits
        if not self.rate_limiter.can_make_request(int(estimated_tokens)):
            wait_time = self.rate_limiter.get_wait_time()
            print(f"⏳ Rate limit approaching, waiting {wait_time} seconds...")
            time.sleep(wait_time)
            
            # Double-check after waiting
            if not self.rate_limiter.can_make_request(int(estimated_tokens)):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
        
        # Optimize prompt for token efficiency
        optimized_prompt = self._optimize_prompt(prompt)
        
        # Format message for chat completion with enhanced system prompt
        messages = [{
            "role": "system",
            "content": """You are Clegora, a friendly and knowledgeable legal assistant specializing in Indian law, particularly Motor Vehicles Act 1988, Consumer Protection Act 2019, and Right to Information Act 2005.

Key principles:
- Provide accurate, well-structured legal guidance in conversational tone
- Always cite specific sections, acts, and legal sources when available
- Use clear, accessible language that anyone can understand
- Be warm and approachable while maintaining legal accuracy
- Give practical, actionable advice when possible
- Distinguish between legal requirements and practical suggestions
- State limitations clearly when context is insufficient

Response approach:
1. Direct, friendly answer to the question
2. Legal basis with specific citations (sections, acts)
3. Practical steps or advice if applicable
4. Additional context or warnings if relevant

Always be helpful, conversational, and focus on what the user can actually do."""
        }, {
            "role": "user",
            "content": optimized_prompt
        }]
        
        # Try models in order of preference
        for model in self._models:
            try:
                response = self._try_model(model, messages)
                if response:
                    # Record successful request
                    self.rate_limiter.record_request(int(estimated_tokens))
                    return response
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
        
        # Final fallback
        return self._fallback_response(prompt)

    def _try_model(self, model: str, messages: List[Dict[str, str]], retries: int = 2) -> Optional[str]:
        """Try a specific model with optimized settings"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.9,
            "stream": False,
            "stop": None
        }
        
        try:
            response = requests.post(
                self._api_url,
                headers=self._headers,
                json=payload,
                timeout=20  # Shorter timeout for faster fallback
            )
            
            if response.status_code == 429:
                # Rate limited
                retry_after = response.headers.get('retry-after', '60')
                wait_time = min(int(retry_after), 300)  # Max 5 min wait
                print(f"Rate limited by API, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return None
            
            elif response.status_code != 200:
                print(f"API error {response.status_code}: {response.text[:200]}")
                return None
            
            result = response.json()
            
            if result.get("choices") and len(result["choices"]) > 0:
                text = result["choices"][0]["message"]["content"]
                return text.strip() if text else None
                
        except requests.exceptions.Timeout:
            print(f"Timeout with model {model}")
        except Exception as e:
            print(f"Error with model {model}: {e}")
        
        return None
    
    def _optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt to use fewer tokens"""
        # Extract key parts if it's a RAG prompt
        if "Context:" in prompt and "Question:" in prompt:
            parts = prompt.split("Question:")
            if len(parts) > 1:
                context_part = parts[0].replace("Context:", "").strip()
                question_part = parts[1].strip()
                
                # Truncate context if too long (keep most relevant parts)
                if len(context_part.split()) > 400:
                    sentences = context_part.split('. ')
                    # Keep first and last parts, which often contain key info
                    if len(sentences) > 6:
                        context_part = '. '.join(sentences[:3] + sentences[-3:])
                
                return f"Context: {context_part}\n\nQuestion: {question_part}\n\nAnswer briefly:"
        
        # For direct questions, add brevity instruction
        if len(prompt.split()) > 200:
            prompt = ' '.join(prompt.split()[:150]) + "..."
        
        return f"{prompt}\n\nProvide a concise answer:"
    
    
    def _fallback_response(self, prompt: str) -> str:
        """Enhanced fallback response based on query topic"""
        prompt_lower = prompt.lower()
        
        # Motor Vehicles Act related queries
        if any(term in prompt_lower for term in ['driving', 'license', 'vehicle', 'traffic', 'fine', 'helmet', 'registration']):
            return """I understand you're asking about motor vehicle regulations. While I'm having technical difficulties accessing my full knowledge base, here's some general guidance:

For driving licenses, vehicle registration, traffic violations, and related matters, the Motor Vehicles Act, 1988 is the primary legislation. You can:

• Visit your local RTO (Regional Transport Office) for official procedures
• Check the official transport department website of your state
• Consult the Motor Vehicles Act, 1988 for detailed provisions

**Important:** For specific legal matters, always consult qualified legal professionals familiar with current Indian motor vehicle law."""
        
        # Consumer Protection related queries
        elif any(term in prompt_lower for term in ['consumer', 'complaint', 'defective', 'refund', 'product', 'service']):
            return """I see you're asking about consumer rights. While I'm experiencing technical difficulties, here's some general guidance:

Under the Consumer Protection Act, 2019, you have rights regarding defective products and services. You can:

• File complaints with Consumer Forums (District, State, or National level)
• Seek compensation for defective goods or deficient services
• Contact consumer helplines in your state

**Important:** For specific consumer disputes, consult qualified legal professionals familiar with current consumer protection law."""
        
        # RTI related queries
        elif any(term in prompt_lower for term in ['rti', 'information', 'government', 'public', 'transparency']):
            return """I understand you're asking about the Right to Information. While I'm having technical difficulties, here's some general guidance:

Under the RTI Act, 2005, you can request information from public authorities. You can:

• File RTI applications with relevant government departments
• Appeal to higher authorities if information is denied
• Contact RTI activists or organizations for guidance

**Important:** For specific RTI matters, consult qualified legal professionals familiar with current information access law."""
        
        # General fallback
        return """I'm currently unable to process your legal query due to technical difficulties. Please try again in a moment.

For immediate assistance with legal matters:
• Consult qualified legal professionals
• Contact relevant government departments
• Visit official legal aid centers

**Disclaimer:** This system provides general legal information only. For specific legal matters, always consult qualified legal professionals familiar with current Indian law."""
    
    @property
    def _llm_type(self) -> str:
        return "groq_llama32"
    
    @property 
    def _identifying_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "api_url": self._api_url,
            "max_tokens": self.max_tokens
        }

class OptimizedLocalEmbeddings(Embeddings):
    """Lightweight local embeddings to save API calls"""
    
    def __init__(self):
        super().__init__()
        # Simple TF-IDF-like embedding for free tier optimization
        self.vocab = {}
        self.embedding_dim = 384
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create simple hash-based embeddings locally"""
        embeddings = []
        for text in texts:
            embedding = self._create_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Create embedding for query"""
        return self._create_embedding(text)
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create deterministic embedding from text"""
        import hashlib
        import re
        
        # Clean and normalize text
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = text.split()
        
        # Create multiple hash values for better distribution
        hashes = []
        for i in range(self.embedding_dim // 32):  # 32 floats per hash
            hash_input = f"{text}_{i}".encode()
            hash_val = hashlib.md5(hash_input).hexdigest()
            hashes.append(hash_val)
        
        # Convert to float embedding
        embedding = []
        for hash_str in hashes:
            for i in range(0, min(32, len(hash_str)), 2):
                hex_val = hash_str[i:i+2]
                float_val = (int(hex_val, 16) / 255.0) - 0.5
                embedding.append(float_val)
                if len(embedding) >= self.embedding_dim:
                    break
        
        # Pad or truncate to exact dimension
        while len(embedding) < self.embedding_dim:
            embedding.append(0.0)
        
        return embedding[:self.embedding_dim]

class GroqLlama32Client:
    """Optimized client for Groq Llama 3.2 free tier"""
    
    def __init__(self):
        try:
            load_dotenv()
            
            # Get Groq API key
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
            
            print("Initializing Groq Llama 3.2 client (Free Tier Optimized)...")
            
            # Initialize optimized LLM
            self.llm = GroqLlama32LLM(
                api_key=self.groq_api_key,
                model_name="llama3-8b-8192"
            )
            print("Language model initialized")
            
            # Use local embeddings to save API calls
            self.embeddings = OptimizedLocalEmbeddings()
            print("Local embeddings initialized")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client()
            
            # Enhanced memory with contextual awareness
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                output_key="answer",  # Specify which output to store in memory
                k=4,  # Small context window (4 exchanges = 8 messages)
                return_messages=True
            )
            
            # Conversation context tracking
            self.conversation_context = {
                "topic_history": [],
                "legal_areas": set(),
                "user_intent": None,
                "context_summary": ""
            }
            
            # Request tracking
            self.request_count = 0
            self.session_start = datetime.now()
            
            print("Groq Llama 3.2 client ready (Free Tier Mode)")
            self.vectorstore = None
            self.retriever = None
            self.qa_chain = None
            
        except Exception as e:
            print(f"Error initializing client: {e}")
            raise

    def initialize_db(self, texts: List[str], metadatas: List[Dict]):
        """Initialize database with token-efficient processing"""
        try:
            print(f"Processing {len(texts)} documents...")
            
            # Clean texts more aggressively to save space
            cleaned_texts = []
            cleaned_metadatas = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    # Truncate very long texts to save tokens
                    cleaned_text = text.strip()
                    if len(cleaned_text.split()) > 300:  # Limit text length
                        words = cleaned_text.split()
                        cleaned_text = ' '.join(words[:300]) + "..."
                    
                    cleaned_texts.append(cleaned_text)
                    cleaned_metadatas.append(metadatas[i])
            
            print(f"Cleaned to {len(cleaned_texts)} valid documents")
            
            if not cleaned_texts:
                raise ValueError("No valid documents found")
            
            # Create vector store (using local embeddings)
            print("Creating vector store...")
            self.vectorstore = Chroma.from_texts(
                texts=cleaned_texts,
                embedding=self.embeddings,
                metadatas=cleaned_metadatas,
                persist_directory="./chroma_db_groq"
            )
            
            # Create retriever with improved relevance filtering
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,  # Get more candidates
                    "score_threshold": 0.3  # Filter out low-relevance results
                }
            )
            
            self._initialize_qa_chain()
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def _initialize_qa_chain(self):
        """Initialize QA chain with enhanced legal prompt template"""
        # Enhanced prompt for better legal reasoning and structure
        prompt_template = """You are Clegora, a knowledgeable and friendly legal assistant specializing in Indian law. Use the provided legal context to give accurate, helpful answers.

LEGAL CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer in a warm, conversational tone like you're helping a friend
- Use the legal context to provide specific, accurate information
- Always cite relevant sections, acts, or legal sources from the context
- Explain legal concepts in simple terms without jargon
- Give practical, actionable advice when possible
- Structure your response clearly: direct answer → legal basis → practical steps
- If the context doesn't fully address the question, say so and provide general guidance
- Use examples to make complex concepts clearer

Provide a helpful, well-structured response:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,  # Re-enable memory
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    async def get_response(self, query: str, conversation_id: str = None) -> Tuple[str, List[str]]:
        """Get response with enhanced contextual awareness"""
        try:
            self.request_count += 1
            
            # Log usage for monitoring
            if self.request_count % 10 == 0:
                elapsed = datetime.now() - self.session_start
                print(f"Session stats: {self.request_count} requests in {elapsed}")
            
            # Analyze and update conversation context
            self._update_conversation_context(query)
            
            # Optimize query length
            if len(query.split()) > 50:
                query = ' '.join(query.split()[:50]) + "..."
            
            # Check if we should use RAG or direct response
            if not self.qa_chain:
                # Direct LLM call with fallback
                response = self._get_fallback_for_query(query)
                return response, []
            
            # Create context-aware query
            enhanced_query = self._enhance_query_with_context(query)
            
            # Use RAG chain with chat history
            chat_history = self.memory.chat_memory.messages
            result = self.qa_chain.invoke({"question": enhanced_query, "chat_history": chat_history})
            
            answer = result.get("answer", "").strip()
            source_docs = result.get("source_documents", [])
            
            # Backend relevance check - filter out irrelevant responses
            if not answer or self._is_response_irrelevant(answer, query):
                # Fallback to general guidance
                fallback_answer = self._get_fallback_for_query(query)
                return fallback_answer, []
            
            # Update context with successful response
            self._update_context_with_response(query, answer)
            
            # Extract sources efficiently
            sources = []
            for doc in source_docs[:3]:  # Show more sources for transparency
                metadata = doc.metadata
                title = metadata.get('title', 'Document')[:60]  # Truncate long titles
                source_type = metadata.get('source', 'legal_doc')
                source = f"{title} ({source_type})"
                if source not in sources:
                    sources.append(source)
            
            return answer, sources
            
        except Exception as e:
            print(f"Error in get_response: {e}")
            # Fallback response
            fallback_answer = self._get_fallback_for_query(query)
            return fallback_answer, []

    def _get_fallback_for_query(self, query: str) -> str:
        """Get appropriate fallback response based on query topic"""
        query_lower = query.lower()
        
        # Motor Vehicles Act related queries
        if any(term in query_lower for term in ['driving', 'license', 'vehicle', 'traffic', 'fine', 'helmet', 'registration', 'motor']):
            return """I'd be happy to help with your motor vehicle question! While I don't have specific details for your exact query right now, here's some general guidance:

**Motor Vehicles Act, 1988** covers most vehicle-related regulations including:
• Driving licenses and their validity
• Vehicle registration procedures
• Traffic violations and penalties
• Safety requirements (helmets, seat belts)
• Insurance requirements

**Next steps:**
• Visit your local RTO office for official procedures
• Check your state transport department's website
• For violations, refer to the specific sections of the Motor Vehicles Act

**Legal Disclaimer:** For specific legal matters, consult qualified legal professionals familiar with current motor vehicle law."""
        
        # Consumer Protection related queries
        elif any(term in query_lower for term in ['consumer', 'complaint', 'defective', 'refund', 'product', 'service', 'warranty']):
            return """I'm here to help with your consumer rights question! Here's some general guidance:

**Consumer Protection Act, 2019** provides comprehensive protection including:
• Rights against defective products and deficient services
• Complaint mechanisms through Consumer Forums
• Compensation for damages and losses
• Product liability provisions

**You can take action by:**
• Filing complaints with District/State/National Consumer Forums
• Seeking refunds, replacements, or compensation
• Contacting consumer helplines

**Legal Disclaimer:** For specific consumer disputes, consult qualified legal professionals familiar with consumer protection law."""
        
        # RTI related queries
        elif any(term in query_lower for term in ['rti', 'information', 'government', 'public', 'transparency', 'application']):
            return """I'm glad to help with your Right to Information query! Here's what you should know:

**Right to Information Act, 2005** empowers citizens to:
• Request information from public authorities
• Get responses within 30 days (48 hours for life/liberty matters)
• Appeal if information is denied or delayed
• Access government records and documents

**How to proceed:**
• File RTI applications with relevant departments
• Pay prescribed fees (₹10 for most applications)
• Appeal to higher authorities if needed

**Legal Disclaimer:** For complex RTI matters, consult legal professionals familiar with information access law."""
        
        # General legal query
        return """I'm here to help with your legal question! While I don't have specific information for your exact query right now, here are some general resources:

**For Indian Legal Matters:**
• Consumer Protection Act, 2019 for consumer rights
• Motor Vehicles Act, 1988 for vehicle-related issues
• Right to Information Act, 2005 for government transparency
• Fundamental Rights under the Constitution

**Recommended actions:**
• Consult qualified legal professionals for specific advice
• Contact relevant government departments
• Visit legal aid centers for assistance

**Legal Disclaimer:** This provides general information only. For specific legal matters, always consult qualified legal professionals familiar with current Indian law."""

    def _is_response_irrelevant(self, answer: str, query: str) -> bool:
        """Backend check to determine if response is irrelevant to query"""
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Check for obvious mismatches
        irrelevant_indicators = [
            "bench member", "disagreement", "majority opinion", "president for resolution",
            "tribunal", "commission member", "judicial", "court procedure"
        ]
        
        # Extract key terms from query
        query_terms = set(query_lower.split())
        important_terms = {"fundamental", "rights", "consumer", "rti", "information", 
                          "complaint", "refund", "defective", "application"}
        
        query_has_important_terms = bool(query_terms.intersection(important_terms))
        
        # If query has important terms but answer contains irrelevant indicators
        if query_has_important_terms and any(indicator in answer_lower for indicator in irrelevant_indicators):
            return True
            
        # Check if answer is too short or generic
        if len(answer.split()) < 10:
            return True
            
        return False

    def _update_conversation_context(self, query: str):
        """Update conversation context with new query"""
        try:
            # Extract legal areas mentioned
            legal_keywords = {
                "consumer": "Consumer Protection",
                "rti": "Right to Information",
                "information": "Right to Information", 
                "complaint": "Consumer Rights",
                "refund": "Consumer Rights",
                "defective": "Consumer Protection",
                "fundamental": "Fundamental Rights",
                "rights": "Legal Rights",
                "driving": "Motor Vehicles",
                "license": "Motor Vehicles",
                "vehicle": "Motor Vehicles",
                "traffic": "Motor Vehicles",
                "fine": "Motor Vehicles",
                "helmet": "Motor Vehicles",
                "registration": "Motor Vehicles",
                "motor": "Motor Vehicles"
            }
            
            query_lower = query.lower()
            for keyword, area in legal_keywords.items():
                if keyword in query_lower:
                    self.conversation_context["legal_areas"].add(area)
            
            # Track topic progression (keep last 3 topics)
            if len(self.conversation_context["topic_history"]) >= 3:
                self.conversation_context["topic_history"].pop(0)
            
            # Simple intent detection
            if any(word in query_lower for word in ["how", "what", "explain"]):
                self.conversation_context["user_intent"] = "information_seeking"
            elif any(word in query_lower for word in ["file", "apply", "submit"]):
                self.conversation_context["user_intent"] = "procedural"
            elif any(word in query_lower for word in ["rights", "can i", "allowed"]):
                self.conversation_context["user_intent"] = "rights_inquiry"
            
            self.conversation_context["topic_history"].append(query[:50])
            
        except Exception as e:
            print(f"Error updating context: {e}")
    
    def _enhance_query_with_context(self, query: str) -> str:
        """Enhance query with conversation context"""
        try:
            context_parts = []
            
            # Add legal area context if available
            if self.conversation_context["legal_areas"]:
                areas = ", ".join(list(self.conversation_context["legal_areas"])[:2])
                context_parts.append(f"Legal context: {areas}")
            
            # Add recent topic context
            if len(self.conversation_context["topic_history"]) > 1:
                recent_topic = self.conversation_context["topic_history"][-2]
                context_parts.append(f"Previous topic: {recent_topic}")
            
            if context_parts:
                context_prefix = " | ".join(context_parts)
                return f"[Context: {context_prefix}] {query}"
            
            return query
            
        except Exception as e:
            print(f"Error enhancing query: {e}")
            return query
    
    def _update_context_with_response(self, query: str, response: str):
        """Update context summary with successful Q&A"""
        try:
            # Create a brief summary of the exchange
            summary_parts = []
            if self.conversation_context["legal_areas"]:
                summary_parts.append(f"Areas: {', '.join(list(self.conversation_context['legal_areas']))[:2]}")
            
            if self.conversation_context["user_intent"]:
                summary_parts.append(f"Intent: {self.conversation_context['user_intent']}")
            
            self.conversation_context["context_summary"] = " | ".join(summary_parts)
            
        except Exception as e:
            print(f"Error updating context summary: {e}")
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context"""
        return {
            "legal_areas": list(self.conversation_context["legal_areas"]),
            "recent_topics": self.conversation_context["topic_history"],
            "user_intent": self.conversation_context["user_intent"],
            "context_summary": self.conversation_context["context_summary"],
            "memory_length": len(self.memory.chat_memory.messages)
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring free tier limits"""
        elapsed = datetime.now() - self.session_start
        return {
            "requests_made": self.request_count,
            "session_duration": str(elapsed),
            "avg_requests_per_hour": round(self.request_count / max(elapsed.total_seconds() / 3600, 0.1), 2),
            "conversation_context": self.get_conversation_context(),
            "rate_limiter_stats": {
                "daily_requests": self.llm.rate_limiter.daily_requests,
                "requests_per_minute": len(self.llm.rate_limiter.request_times)
            }
        }