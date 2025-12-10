"""
Django views for 3GPP Knowledge Chatbot.
Handles chat UI rendering and API endpoints for RAG queries.
"""
import sys
import os
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import centralized logging with custom levels
from logging_config import setup_centralized_logging, get_logger, CRITICAL, ERROR, MAJOR, MINOR, DEBUG

# Initialize logging
setup_centralized_logging()
logger = get_logger('Chatbot')

# Import RAG system V3 (hybrid retrieval)
try:
    from rag_system_v3 import create_rag_system_v3
    RAG_AVAILABLE = True
    logger.log(MAJOR, "RAG system V3 module imported successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    logger.log(ERROR, f"Failed to import RAG system V3: {e}")


class RAGManager:
    """
    Singleton manager for RAG system lifecycle.
    Handles initialization, queries, and cleanup.
    """
    _instance = None
    _rag_system = None
    _initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize_rag(self):
        """Initialize RAG system with API keys from environment"""
        if not self._initialized and RAG_AVAILABLE:
            try:
                logger.log(MAJOR, "Initializing RAG system...")

                # Get API key from environment
                claude_api_key = os.getenv('CLAUDE_API_KEY')
                if not claude_api_key:
                    logger.log(ERROR, "CLAUDE_API_KEY environment variable not set")
                    return False

                local_llm_url = "http://192.168.1.14:11434/api/chat"
                self._rag_system = create_rag_system_v3(claude_api_key, local_llm_url)
                self._initialized = True
                logger.log(MAJOR, "RAG system V3 initialized successfully")
                return True

            except Exception as e:
                logger.log(CRITICAL, f"Failed to initialize RAG system: {e}")
                import traceback
                traceback.print_exc()

        elif self._initialized:
            logger.log(DEBUG, "RAG system already initialized")
            return True

        logger.log(ERROR, "RAG system not available")
        return False

    def query(self, question, max_depth=1, max_chunks_per_level=5, model="claude"):
        """Execute RAG query and return response"""
        if self._rag_system and self._initialized:
            try:
                logger.log(MINOR, f"RAG query: {question[:60]}... (model: {model})")
                response = self._rag_system.query(question, model)
                response_length = len(response.answer) if response else 0
                logger.log(MAJOR, f"RAG response generated - {response_length} chars")
                return response

            except Exception as e:
                logger.log(ERROR, f"RAG query error: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            logger.log(ERROR, f"RAG system not ready: initialized={self._initialized}")
        return None

    def is_ready(self):
        return self._initialized and self._rag_system is not None

    def close(self):
        """Clean up RAG system resources"""
        if self._rag_system:
            logger.log(MAJOR, "Closing RAG system")
            self._rag_system.close()
            self._rag_system = None
            self._initialized = False
            logger.log(MINOR, "RAG system closed")


# Initialize RAG system at module level
rag_manager = RAGManager.get_instance()
logger.log(MAJOR, "Starting RAG system initialization")
rag_initialized = rag_manager.initialize_rag()
logger.log(MAJOR, f"RAG initialization complete - Status: {rag_initialized}")


def chat_view(request):
    """Render chat UI template"""
    return render(request, 'chatbot/chat.html')


@method_decorator(csrf_exempt, name='dispatch')
class ChatAPIView(View):
    """
    API endpoint for chat interactions.
    Receives user messages and returns RAG-generated responses.
    """

    def post(self, request):
        """Handle chat POST request"""
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            selected_model = data.get('model', 'deepseek-r1:14b')

            logger.log(MINOR, f"Chat request - Model: {selected_model}")
            logger.log(DEBUG, f"Message: {user_message[:80]}...")

            # Validate message
            if not user_message:
                logger.log(ERROR, "Empty message received")
                return JsonResponse({'error': 'Message cannot be empty'}, status=400)

            # Process and respond
            bot_response = self.get_bot_response(user_message, selected_model)
            logger.log(MAJOR, f"Chat response - {len(bot_response)} chars")

            return JsonResponse({
                'user_message': user_message,
                'bot_response': bot_response,
                'model_used': selected_model
            })

        except json.JSONDecodeError:
            logger.log(ERROR, "Invalid JSON in request")
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            logger.log(CRITICAL, f"Chat API error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    def get_bot_response(self, user_message, model="claude"):
        """Generate response using RAG system or fallback"""
        backend_model = self.map_ui_model_to_backend(model)
        logger.log(MINOR, f"Processing with model: {backend_model}")

        # Try RAG system if available
        if RAG_AVAILABLE and rag_manager.is_ready():
            try:
                logger.log(DEBUG, "Querying RAG system")
                rag_response = rag_manager.query(
                    question=user_message,
                    max_depth=1,
                    max_chunks_per_level=5,
                    model=backend_model
                )

                if rag_response and rag_response.answer:
                    logger.log(MAJOR, f"RAG response successful with {backend_model}")
                    return rag_response.answer
                else:
                    logger.log(ERROR, "RAG returned empty response")

            except Exception as e:
                logger.log(ERROR, f"RAG system error: {e}")
                import traceback
                traceback.print_exc()

        # Fallback to simple responses
        logger.log(MINOR, "Using simple response fallback")
        return self.get_simple_response(user_message, model)

    def get_model_display_name(self, model):
        """Get human-readable model name"""
        model_names = {
            "claude": "Claude (Anthropic)",
            "deepseek": "DeepSeek R1:7B",
            "deepseek-r1:14b": "DeepSeek R1:14B",
            "deepseek-r1:7b": "DeepSeek R1:7B",
            "gemma3:12b": "Gemma 3:12B",
            "mistral:7b": "Mistral:7B",
            "llama3.1:8b": "Llama 3.1:8B"
        }
        return model_names.get(model, model.title())

    def map_ui_model_to_backend(self, ui_model):
        """Map UI model selection to backend model identifier"""
        model_mapping = {
            "claude": "claude",
            "deepseek-r1:7b": "deepseek-r1:7b",
            "deepseek-r1:14b": "deepseek-r1:14b",
            "gemma3:12b": "gemma3:12b",
            "mistral:7b": "mistral:7b",
            "llama3.1:8b": "llama3.1:8b"
        }
        return model_mapping.get(ui_model, "claude")

    def get_simple_response(self, user_message, model="claude"):
        """Generate simple fallback response when RAG unavailable"""
        user_message_lower = user_message.lower()
        model_display = self.get_model_display_name(model)

        logger.log(DEBUG, f"Simple response for: {user_message_lower[:30]}...")

        if 'hello' in user_message_lower or 'hi' in user_message_lower:
            response = f"Hello! I'm your 3GPP knowledge assistant powered by {model_display}. Ask me about 5G, network functions, procedures, or technical specifications."
        elif 'help' in user_message_lower:
            response = f"I can help you with 3GPP technical questions about 5G architecture, network functions (AMF, SMF, UPF, etc.), procedures, and specifications using {model_display} AI model. Just ask your question!"
        elif 'what' in user_message_lower and ('you' in user_message_lower or 'this' in user_message_lower):
            response = f"I'm a 3GPP knowledge chatbot powered by RAG (Retrieval Augmented Generation) using {model_display} AI model. I can answer technical questions about 5G system architecture and specifications."
        else:
            if not rag_manager.is_ready():
                response = f"I'm a 3GPP knowledge assistant using {model_display}, but my RAG system is currently not available. Please check the system configuration or try asking about basic 5G concepts."
            else:
                response = f"I couldn't find specific information about that topic in my knowledge base using {model_display}. Try asking about 5G network functions, procedures, or technical specifications."

        return response
