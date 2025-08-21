import firebase_admin
from firebase_admin import credentials, firestore, auth
from typing import Dict, List, Optional, Any
import os
import json
from datetime import datetime
import asyncio
from functools import wraps
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class FirebaseService:
    """Firebase service for user authentication and chat context storage"""
    
    def __init__(self):
        self.db = None
        self.app = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase with service account credentials"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # Look for service account key file
                service_account_paths = [
                    os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH'),
                    'firebase-service-account.json',
                    'config/firebase-service-account.json',
                    os.path.join(os.path.dirname(__file__), '..', '..', 'firebase-service-account.json')
                ]
                
                service_account_path = None
                for path in service_account_paths:
                    if path and os.path.exists(path):
                        service_account_path = path
                        break
                
                if service_account_path:
                    # Initialize with service account
                    cred = credentials.Certificate(service_account_path)
                    self.app = firebase_admin.initialize_app(cred)
                    print(f"✅ Firebase initialized with service account: {service_account_path}")
                else:
                    # Try to initialize with environment variables
                    firebase_config = {
                        "type": os.getenv('FIREBASE_TYPE', 'service_account'),
                        "project_id": os.getenv('FIREBASE_PROJECT_ID'),
                        "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
                        "private_key": os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
                        "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
                        "client_id": os.getenv('FIREBASE_CLIENT_ID'),
                        "auth_uri": os.getenv('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                        "token_uri": os.getenv('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                        "auth_provider_x509_cert_url": os.getenv('FIREBASE_AUTH_PROVIDER_CERT_URL'),
                        "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL')
                    }
                    
                    # Check if required fields are present
                    if firebase_config['project_id'] and firebase_config['private_key'] and firebase_config['client_email']:
                        cred = credentials.Certificate(firebase_config)
                        self.app = firebase_admin.initialize_app(cred)
                        print("✅ Firebase initialized with environment variables")
                    else:
                        print("⚠️ Firebase credentials not found. Please configure Firebase.")
                        print("Required: FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL")
                        return
            else:
                self.app = firebase_admin.get_app()
                print("✅ Firebase already initialized")
            
            # Initialize Firestore
            self.db = firestore.client()
            print("✅ Firestore client initialized")
            
        except Exception as e:
            print(f"❌ Error initializing Firebase: {e}")
            self.db = None
            self.app = None
    
    def is_initialized(self) -> bool:
        """Check if Firebase is properly initialized"""
        return self.db is not None
    
    async def verify_user_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """Verify Firebase ID token and return user info"""
        if not self.is_initialized():
            raise Exception("Firebase not initialized")
        
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(id_token)
            user_id = decoded_token['uid']
            
            return {
                'uid': user_id,
                'email': decoded_token.get('email'),
                'name': decoded_token.get('name'),
                'email_verified': decoded_token.get('email_verified', False)
            }
        except Exception as e:
            print(f"❌ Error verifying token: {e}")
            return None
    
    async def create_user_profile(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """Create or update user profile in Firestore"""
        if not self.is_initialized():
            raise Exception("Firebase not initialized")
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            
            profile_data = {
                'email': user_data.get('email'),
                'name': user_data.get('name'),
                'created_at': datetime.now(),
                'last_login': datetime.now(),
                'total_chats': 0,
                'preferences': {
                    'theme': 'light',
                    'language': 'en'
                }
            }
            
            # Check if user already exists
            user_doc = user_ref.get()
            if user_doc.exists:
                # Update existing user
                profile_data.pop('created_at')  # Don't update creation time
                existing_data = user_doc.to_dict()
                profile_data['total_chats'] = existing_data.get('total_chats', 0)
                user_ref.update(profile_data)
                print(f"✅ Updated user profile: {user_id}")
            else:
                # Create new user
                user_ref.set(profile_data)
                print(f"✅ Created user profile: {user_id}")
            
            return True
        except Exception as e:
            print(f"❌ Error creating user profile: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from Firestore"""
        if not self.is_initialized():
            raise Exception("Firebase not initialized")
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                return user_doc.to_dict()
            return None
        except Exception as e:
            print(f"❌ Error getting user profile: {e}")
            return None
    
    async def save_chat_context(self, user_id: str, conversation_id: str, context_data: Dict[str, Any]) -> bool:
        """Save chat context to Firestore"""
        if not self.is_initialized():
            raise Exception("Firebase not initialized")
        
        try:
            # Save to user's chat contexts collection
            context_ref = self.db.collection('users').document(user_id).collection('chat_contexts').document(conversation_id)
            
            context_data.update({
                'updated_at': datetime.now(),
                'user_id': user_id,
                'conversation_id': conversation_id
            })
            
            context_ref.set(context_data, merge=True)
            
            # Update user's total chat count
            user_ref = self.db.collection('users').document(user_id)
            user_ref.update({
                'last_activity': datetime.now(),
                'total_chats': firestore.Increment(1)
            })
            
            print(f"✅ Saved chat context: {conversation_id} for user: {user_id}")
            return True
        except Exception as e:
            print(f"❌ Error saving chat context: {e}")
            return False
    
    async def get_chat_context(self, user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get chat context from Firestore"""
        if not self.is_initialized():
            raise Exception("Firebase not initialized")
        
        try:
            context_ref = self.db.collection('users').document(user_id).collection('chat_contexts').document(conversation_id)
            context_doc = context_ref.get()
            
            if context_doc.exists:
                return context_doc.to_dict()
            return None
        except Exception as e:
            print(f"❌ Error getting chat context: {e}")
            return None
    
    async def get_user_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's recent chat conversations"""
        if not self.is_initialized():
            raise Exception("Firebase not initialized")
        
        try:
            contexts_ref = (self.db.collection('users')
                          .document(user_id)
                          .collection('chat_contexts')
                          .order_by('updated_at', direction=firestore.Query.DESCENDING)
                          .limit(limit))
            
            contexts = contexts_ref.stream()
            chat_history = []
            
            for context in contexts:
                context_data = context.to_dict()
                chat_history.append({
                    'conversation_id': context.id,
                    'title': context_data.get('title', 'Untitled Chat'),
                    'updated_at': context_data.get('updated_at'),
                    'message_count': context_data.get('message_count', 0),
                    'last_message': context_data.get('last_message', '')
                })
            
            return chat_history
        except Exception as e:
            print(f"❌ Error getting chat history: {e}")
            return []
    
    async def delete_chat_context(self, user_id: str, conversation_id: str) -> bool:
        """Delete a specific chat context"""
        if not self.is_initialized():
            raise Exception("Firebase not initialized")
        
        try:
            context_ref = self.db.collection('users').document(user_id).collection('chat_contexts').document(conversation_id)
            context_ref.delete()
            
            print(f"✅ Deleted chat context: {conversation_id} for user: {user_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting chat context: {e}")
            return False
    
    async def update_chat_metadata(self, user_id: str, conversation_id: str, 
                                 title: str = None, last_message: str = None) -> bool:
        """Update chat metadata (title, last message, etc.)"""
        if not self.is_initialized():
            raise Exception("Firebase not initialized")
        
        try:
            context_ref = self.db.collection('users').document(user_id).collection('chat_contexts').document(conversation_id)
            
            update_data = {
                'updated_at': datetime.now()
            }
            
            if title:
                update_data['title'] = title
            if last_message:
                update_data['last_message'] = last_message[:200]  # Truncate for storage efficiency
                update_data['message_count'] = firestore.Increment(1)
            
            context_ref.update(update_data)
            return True
        except Exception as e:
            print(f"❌ Error updating chat metadata: {e}")
            return False

# Global Firebase service instance
firebase_service = FirebaseService()

def require_auth(f):
    """Decorator to require authentication for endpoints"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        # This decorator can be used with FastAPI dependency injection
        # The actual auth check will be implemented in the endpoints
        return await f(*args, **kwargs)
    return decorated_function
