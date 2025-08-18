import sys
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json

# Add parent directory to path to import RAG system
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

try:
    from rag_system import create_rag_system
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

class RAGManager:
    _instance = None
    _rag_system = None
    _initialized = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize_rag(self):
        if not self._initialized and RAG_AVAILABLE:
            try:
                claude_api_key = "..."
                if claude_api_key:
                    print("Initializing RAG system...")
                    self._rag_system = create_rag_system(claude_api_key)
                    self._initialized = True
                    print("RAG system initialized successfully!")
                    return True
            except Exception as e:
                print(f"Failed to initialize RAG system: {e}")
                import traceback
                traceback.print_exc()
        elif self._initialized:
            return True
        return False
    
    def query(self, question, max_depth=1, max_chunks_per_level=5):
        if self._rag_system and self._initialized:
            try:
                print(f"RAG query: {question[:50]}...")
                response = self._rag_system.query(question, max_depth, max_chunks_per_level)
                print(f"RAG response length: {len(response.answer) if response else 0}")
                return response
            except Exception as e:
                print(f"RAG query error: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"RAG system not available: initialized={self._initialized}, system={self._rag_system is not None}")
        return None
    
    def is_ready(self):
        return self._initialized and self._rag_system is not None
    
    def close(self):
        if self._rag_system:
            self._rag_system.close()
            self._rag_system = None
            self._initialized = False

# Initialize RAG system at module level
rag_manager = RAGManager.get_instance()
rag_initialized = rag_manager.initialize_rag()

def chat_view(request):
    return render(request, 'chatbot/chat.html')

@method_decorator(csrf_exempt, name='dispatch')
class ChatAPIView(View):        
    def post(self, request):
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            
            if not user_message:
                return JsonResponse({'error': 'Message cannot be empty'}, status=400)
            
            bot_response = self.get_bot_response(user_message)
            
            return JsonResponse({
                'user_message': user_message,
                'bot_response': bot_response
            })
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    def get_bot_response(self, user_message):
        print(f"Processing message: {user_message}")
        print(f"RAG Available: {RAG_AVAILABLE}")
        print(f"RAG Ready: {rag_manager.is_ready()}")
        
        # First try RAG system if available and initialized
        if RAG_AVAILABLE and rag_manager.is_ready():
            try:
                rag_response = rag_manager.query(
                    question=user_message,
                    max_depth=1,
                    max_chunks_per_level=5
                )
                
                if rag_response and rag_response.answer:
                    print("Using RAG response")
                    return rag_response.answer
                else:
                    print("RAG returned empty response")
                    
            except Exception as e:
                print(f"RAG system error: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to simple responses if RAG is not available
        print("Falling back to simple response")
        return self.get_simple_response(user_message)
    
    def get_simple_response(self, user_message):
        user_message_lower = user_message.lower()
        
        if 'hello' in user_message_lower or 'hi' in user_message_lower:
            return "Hello! I'm your 3GPP knowledge assistant. Ask me about 5G, network functions, procedures, or technical specifications."
        elif 'help' in user_message_lower:
            return "I can help you with 3GPP technical questions about 5G architecture, network functions (AMF, SMF, UPF, etc.), procedures, and specifications. Just ask your question!"
        elif 'what' in user_message_lower and ('you' in user_message_lower or 'this' in user_message_lower):
            return "I'm a 3GPP knowledge chatbot powered by RAG (Retrieval Augmented Generation). I can answer technical questions about 5G system architecture and specifications."
        else:
            if not self.rag_initialized:
                return "I'm a 3GPP knowledge assistant, but my RAG system is currently not available. Please check the system configuration or try asking about basic 5G concepts."
            else:
                return "I couldn't find specific information about that topic in my knowledge base. Try asking about 5G network functions, procedures, or technical specifications."
