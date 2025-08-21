# Legal Chatbot Deployment Guide

## 🚀 Free Deployment Setup

Your legal chatbot is now ready for deployment with ChromaDB bundled with the backend. Here's how to deploy it for free:

## Backend Deployment (Render.com)

### 1. Prepare Your Repository
```bash
# Make sure all files are committed
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Deploy to Render
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Use these settings:
   - **Name**: `legal-assistant-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Instance Type**: `Free`

### 3. Environment Variables
Add these in Render dashboard:
```
GROQ_API_KEY=your_groq_api_key_here
CHROMA_DB_PATH=./chroma_db
DEBUG=false
```

## Frontend Deployment (Netlify)

### 1. Deploy to Netlify
1. Go to [netlify.com](https://netlify.com) and sign up
2. Drag and drop your `frontend/build` folder to Netlify
3. Or connect your GitHub repo and set:
   - **Build command**: `cd frontend && npm install && npm run build`
   - **Publish directory**: `frontend/build`

### 2. Update API URL
After backend deployment, update `netlify.toml`:
```toml
[[redirects]]
  from = "/api/*"
  to = "https://your-backend-url.onrender.com/api/:splat"
  status = 200
  force = true
```

## Alternative Free Options

### Railway (Backend)
1. Go to [railway.app](https://railway.app)
2. Deploy from GitHub
3. Add environment variables
4. Uses `Procfile` automatically

### Vercel (Frontend)
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repo
3. Set build settings:
   - **Framework**: Create React App
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`

## 📁 Files Created for Deployment

### Backend Files:
- `app.py` - Production entry point with ChromaDB bundling
- `Procfile` - For Railway/Heroku deployment
- `runtime.txt` - Python version specification
- `render.yaml` - Render.com configuration
- `.env.production` - Environment variables template

### Frontend Files:
- `netlify.toml` - Netlify configuration with API redirects
- `frontend/build/` - Production build (ready to deploy)

## 🔧 Key Features Bundled:

### Backend:
- ✅ ChromaDB bundled with persistent storage
- ✅ Groq API integration with rate limiting
- ✅ Safety validation system
- ✅ Firebase authentication support
- ✅ CORS configured for frontend
- ✅ Production error handling

### Frontend:
- ✅ React production build optimized
- ✅ Firebase authentication UI
- ✅ Chat interface with real-time responses
- ✅ Responsive design with Tailwind CSS

## 🚀 Quick Deploy Commands

```bash
# Backend (after setting up Render)
git push origin main

# Frontend (manual Netlify deploy)
# Just drag the frontend/build folder to Netlify dashboard
```

## 📊 Free Tier Limits:
- **Render**: 750 hours/month, sleeps after 15min inactivity
- **Netlify**: 100GB bandwidth, 300 build minutes/month
- **Railway**: $5 credit/month (no sleep)
- **Vercel**: 100GB bandwidth, unlimited builds

## 🔑 Next Steps:
1. Get your Groq API key from [console.groq.com](https://console.groq.com)
2. Deploy backend to Render
3. Deploy frontend to Netlify
4. Update API URLs in netlify.toml
5. Test the deployed application

Your legal chatbot is production-ready with all safety features and optimizations!
