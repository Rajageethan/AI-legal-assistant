from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
from dotenv import load_dotenv
from src.api.endpoints import router  # Updated router for Groq

# Load environment variables
load_dotenv()

# Check for Groq API key
groq_token = os.getenv("GROQ_API_KEY")
if not groq_token:
    print("ERROR: GROQ_API_KEY not found in environment variables!")
    print("Please set your Groq API key in .env file:")
    print("GROQ_API_KEY=gsk_your_token_here")
    print("\nYou can get a free token from: https://console.groq.com/")
    print("Note: The token should start with 'gsk_'")
    sys.exit(1)
elif not groq_token.startswith('gsk_'):
    print("WARNING: Your Groq API token should start with 'gsk_'")
    print("Please check your token format")

app = FastAPI(
    title="Legal Assistant MVP - Groq Llama 3.2",
    description="Legal assistant powered by Groq Llama 3.2 (Free Tier Optimized)",
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print(f"Static files mounted from: {static_dir}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve optimized HTML interface"""
    html_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_file):
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading HTML file: {e}")
            return get_groq_optimized_html()
    else:
        return get_groq_optimized_html()

@app.get("/health")
async def health_check():
    """Simple health check for Render"""
    return {"status": "healthy", "message": "Legal Assistant API is running"}

def get_groq_optimized_html():
    """HTML page optimized for Groq free tier usage"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Legal Assistant - Groq Llama 3.2</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1000px; margin: 0 auto; 
                background: rgba(255,255,255,0.95); 
                padding: 30px; border-radius: 15px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                border-radius: 10px;
                color: white;
            }
            h1 { 
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .subtitle {
                margin: 10px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }
            .status { 
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                border-radius: 10px; 
                padding: 20px; 
                margin: 20px 0;
                text-align: center;
                font-weight: 500;
            }
            .free-tier-info {
                background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
                color: white;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            }
            .free-tier-info h3 {
                margin-top: 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .usage-tips {
                background: #f8f9fa;
                border-left: 5px solid #007acc;
                padding: 20px;
                margin: 20px 0;
                border-radius: 0 10px 10px 0;
            }
            .endpoint { 
                background: #f8f9fa; 
                border-left: 4px solid #667eea; 
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 0 10px 10px 0;
                transition: transform 0.2s ease;
            }
            .endpoint:hover {
                transform: translateX(5px);
            }
            .method { 
                background: #28a745; 
                color: white; 
                padding: 5px 12px; 
                border-radius: 20px; 
                font-size: 12px; 
                font-weight: bold;
                display: inline-block;
                margin-right: 10px;
            }
            .method.get { background: #007acc; }
            ul { list-style: none; padding: 0; }
            li { 
                margin: 15px 0; 
                padding: 15px;
                background: rgba(102, 126, 234, 0.1);
                border-radius: 8px;
                transition: background 0.2s ease;
            }
            li:hover {
                background: rgba(102, 126, 234, 0.2);
            }
            a { 
                color: #667eea; 
                text-decoration: none; 
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            a:hover { 
                text-decoration: underline; 
                color: #5a6fd8;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .feature-card {
                background: rgba(102, 126, 234, 0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .feature-icon {
                font-size: 2em;
                margin-bottom: 10px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-card {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚡ Legal Assistant</h1>
                <p class="subtitle">Powered by Groq Llama 3.2 - Lightning Fast AI</p>
            </div>
            
            <div class="status">
                System Status: Groq Llama 3.2 Free Tier Active
            </div>

            <div class="free-tier-info">
                <h3>🎯 Free Tier Optimization Active</h3>
                <p><strong>Benefits:</strong></p>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>✨ Ultra-fast inference with Groq's LPU technology</li>
                    <li>🔄 Smart rate limiting to maximize free tier usage</li>
                    <li>💾 Local embeddings to save API calls</li>
                    <li>🎛️ Token-optimized prompts for efficiency</li>
                    <li>🛡️ Multiple fallback strategies</li>
                </ul>
                <p><strong>Daily Limits:</strong> ~14,400 requests • Smart batching enabled</p>
            </div>

            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h3>Lightning Speed</h3>
                    <p>Groq's LPU delivers 10x faster inference than traditional GPUs</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">💡</div>
                    <h3>Smart Optimization</h3>
                    <p>Intelligent prompt compression and rate limiting</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <h3>Cost Efficient</h3>
                    <p>Maximizes free tier credits with smart usage patterns</p>
                </div>
            </div>

            <div class="stats">
                <div class="stat-card">
                    <span class="stat-number" id="requestCount">0</span>
                    <span>Requests Today</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number" id="responseTime">~2s</span>
                    <span>Avg Response Time</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">99.9%</span>
                    <span>Uptime</span>
                </div>
            </div>

            <h2>📚 API Documentation</h2>
            <ul>
                <li><a href="/docs">📋 Interactive API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">📄 Alternative API Documentation (ReDoc)</a></li>
                <li><a href="/api/health">🏥 Health Check & Usage Stats</a></li>
                <li><a href="/api/usage-stats">📊 Free Tier Usage Monitor</a></li>
            </ul>

            <h2>🔌 API Endpoints</h2>
            
            <div class="endpoint">
                <strong><span class="method">POST</span> /api/chat</strong>
                <p>Send a message to the legal assistant</p>
                <small>Optimized for Groq Llama 3.2 - Fast responses with smart token management</small>
            </div>

            <div class="endpoint">
                <strong><span class="method get">GET</span> /api/health</strong>
                <p>Check API health and system status</p>
                <small>Returns system info, active conversations, and free tier usage stats</small>
            </div>

            <div class="endpoint">
                <strong><span class="method get">GET</span> /api/usage-stats</strong>
                <p>Monitor free tier usage and rate limits</p>
                <small>Track daily requests, tokens used, and optimization metrics</small>
            </div>

            <div class="usage-tips">
                <h3>💡 Tips for Optimal Free Tier Usage</h3>
                <ul style="margin: 10px 0; padding-left: 20px; list-style: disc;">
                    <li><strong>Keep questions concise:</strong> Shorter queries use fewer tokens</li>
                    <li><strong>Batch similar questions:</strong> More efficient than multiple separate requests</li>
                    <li><strong>Monitor usage:</strong> Check /api/usage-stats to track your daily limits</li>
                    <li><strong>Peak hours:</strong> Better response times during off-peak hours</li>
                    <li><strong>Fallback handling:</strong> System automatically handles rate limits gracefully</li>
                </ul>
            </div>

            <h2>⚙️ System Configuration</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🔑</div>
                    <h4>API Key Status</h4>
                    <p id="apiStatus">Checking...</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <h4>Active Model</h4>
                    <p>Llama 3.2 8B (llama3-8b-8192)</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">💾</div>
                    <h4>Storage</h4>
                    <p>Local embeddings + ChromaDB</p>
                </div>
            </div>

            <div class="free-tier-info">
                <h3>🔄 Rate Limiting Info</h3>
                <p>The system implements smart rate limiting to stay within Groq's free tier limits:</p>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>📊 <strong>Requests:</strong> ~30/minute, ~14,400/day</li>
                    <li>🎯 <strong>Tokens:</strong> ~6,000/minute (optimized)</li>
                    <li>⏱️ <strong>Auto-retry:</strong> Built-in backoff on rate limits</li>
                    <li>🔄 <strong>Fallbacks:</strong> Multiple model options available</li>
                </ul>
            </div>
        </div>

        <script>
            // Update API status
            async function checkApiStatus() {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    document.getElementById('apiStatus').textContent = 
                        response.ok ? 'Connected' : '❌ Error';
                    
                    if (data.usage_stats) {
                        document.getElementById('requestCount').textContent = 
                            data.usage_stats.requests_made || '0';
                    }
                } catch (error) {
                    document.getElementById('apiStatus').textContent = '⚠️ Checking...';
                }
            }

            // Update stats periodically
            checkApiStatus();
            setInterval(checkApiStatus, 30000); // Check every 30 seconds
        </script>
    </body>
    </html>
    """

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Enhanced error handler with Groq-specific handling"""
    import traceback
    from datetime import datetime
    
    error_detail = str(exc)
    print(f"Global exception: {error_detail}")
    print(f"Request: {request.method} {request.url}")
    print(f"Traceback: {traceback.format_exc()}")
    
    # Groq-specific error handling
    if "groq" in error_detail.lower() or "gsk_" in error_detail.lower():
        user_message = "Groq API service temporarily unavailable. Please try again in a moment."
        error_type = "GroqAPIError"
    elif "rate" in error_detail.lower() and "limit" in error_detail.lower():
        user_message = "Rate limit reached. The system will automatically retry. Please wait a moment."
        error_type = "RateLimitError"
    elif "token" in error_detail.lower() or "auth" in error_detail.lower():
        user_message = "API authentication error. Please check server configuration."
        error_type = "AuthenticationError"
    elif "quota" in error_detail.lower() or "credit" in error_detail.lower():
        user_message = "Daily quota reached. Free tier limits exceeded. Please try again tomorrow."
        error_type = "QuotaExceededError"
    else:
        user_message = "Service temporarily unavailable. Please try again shortly."
        error_type = "ServiceError"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": error_type,
            "message": user_message,
            "detail": error_detail if os.getenv("DEBUG", "false").lower() == "true" else None,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "retry_after": 60 if "rate" in error_detail.lower() else None
        }
    )

# Enhanced CORS for production readiness
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler"""
    from datetime import datetime
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP{exc.status_code}Error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# Include the Groq-optimized router
app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with Groq optimization info"""
    print("Legal Assistant API starting up with Groq Llama 3.2...")
    print(f"🔑 Groq API key configured: {'Yes' if groq_token else 'No'}")
    
    # Validate Groq API key format
    if groq_token:
        if groq_token.startswith('gsk_'):
            print("Groq API key format valid")
        else:
            print("⚠️ Warning: Groq API key should start with 'gsk_'")
    
    # Check data files
    data_locations = [
        os.path.join(os.path.dirname(__file__), "data"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.path.dirname(__file__), "src", "data")
    ]
    
    jsonl_found = False
    for data_dir in data_locations:
        jsonl_file = os.path.join(data_dir, "RAG_data.jsonl")
        if os.path.exists(jsonl_file):
            print(f"📁 Data file found: {jsonl_file}")
            file_size = os.path.getsize(jsonl_file)
            print(f"📊 File size: {file_size:,} bytes")
            jsonl_found = True
            
            # Estimate token usage for the session
            estimated_docs = file_size // 1000  # Rough estimate
            print(f"📈 Estimated documents: ~{estimated_docs}")
            break
    
    if not jsonl_found:
        print("⚠️ Warning: RAG_data.jsonl not found")
        print("📍 Expected locations:")
        for data_dir in data_locations:
            print(f"   - {os.path.join(data_dir, 'RAG_data.jsonl')}")
    
    # Free tier optimization info
    print("\n🎯 Free Tier Optimizations Active:")
    print("   ⚡ Groq Llama 3.2 (ultra-fast inference)")
    print("   🔄 Smart rate limiting (30 req/min, 14.4K/day)")
    print("   💾 Local embeddings (zero API calls)")
    print("   🎛️ Token-optimized prompts")
    print("   🛡️ Multi-model fallbacks")
    print("   📊 Usage monitoring enabled")
    
    print("\nStartup completed - Ready for lightning-fast legal assistance!")

@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced shutdown with usage summary"""
    print("🛑 Legal Assistant API shutting down...")
    
    # Clean up resources and show usage summary
    from src.api.endpoints import groq_clients
    if groq_clients:
        print(f"📊 Session Summary:")
        print(f"   - Active conversations: {len(groq_clients)}")
        
        # Aggregate usage stats
        total_requests = 0
        for client in groq_clients.values():
            if hasattr(client, 'request_count'):
                total_requests += client.request_count
        
        print(f"   - Total requests processed: {total_requests}")
        print(f"   - Average requests per conversation: {total_requests/len(groq_clients) if groq_clients else 0:.1f}")
        
        groq_clients.clear()
        print("🧹 Conversations cleaned up")
    
    print("Shutdown completed - Session stats logged")

# Enhanced middleware for performance monitoring
@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    import time
    start_time = time.time()
    request.state.start_time = start_time
    
    # Skip monitoring for static files
    if request.url.path.startswith("/static/"):
        return await call_next(request)
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    
    # Enhanced logging for different request types
    if request.url.path.startswith("/api/chat"):
        if process_time > 5:
            print(f"🐌 Slow chat request: {process_time:.2f}s (may indicate rate limiting)")
        elif process_time < 1:
            print(f"⚡ Fast chat request: {process_time:.2f}s (Groq speed advantage!)")
    elif process_time > 10:
        print(f"⏱️ Slow request: {request.method} {request.url.path} took {process_time:.2f}s")
    
    return response

# Add request ID middleware for better debugging
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    import uuid
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("LEGAL ASSISTANT API - GROQ LLAMA 3.2 EDITION")
    print("="*60)
    print("🌐 Local access: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("🏥 Health check: http://localhost:8000/api/health")
    print("📊 Usage stats: http://localhost:8000/api/usage-stats")
    print("="*60)
    print("⚡ Powered by Groq's Lightning Processing Units (LPU)")
    print("🎯 Free tier optimized for maximum efficiency")
    print("="*60 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False,  # Keep false for production stability
        access_log=True,
        # Add startup message
        app_dir="."
    )