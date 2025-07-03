from config import settings
import google.generativeai as genai
from typing import List, Dict, Optional

class GeminiChat:
    _model = None  # Singleton-like model instance

    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        if GeminiChat._model is None:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            GeminiChat._model = genai.GenerativeModel(model_name)
        self.model = GeminiChat._model
        self.prompt_template = """You are a helpful assistant.
        Use the following document context and chat history to answer the user question accurately.
        You must rpely general knowledge based questions in fun and interactively engaging way.
        Context:
        {context}

        Chat History:
        {history}

        User Question:
        {user_input}"""

    def generate_response(self, user_input: str, context: Optional[str] = None, 
                        chat_history: Optional[List[Dict]] = None) -> str:
        """
        Generates a response from Gemini using context and sliding chat history.
        
        Args:
            user_input: Current user query.
            context: Retrieved context from Qdrant (RAG chunks).
            chat_history: Sliding window of recent chat turns.
        
        Returns:
            str: LLM-generated response.
        
        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If Gemini API call fails.
        """
        try:
            # Validate inputs
            if not user_input:
                raise ValueError("User input cannot be empty")

            # Format chat history
            history_str = ""
            if chat_history:
                for turn in chat_history:
                    role = "User" if turn["role"].upper() == "USER" else "Assistant"
                    history_str += f"{role}: {turn['content']}\n"

            # Build prompt
            prompt = self.prompt_template.format(
                context=context or "No context provided",
                history=history_str or "No history yet.",
                user_input=user_input
            )

            # Generate response
            response = self.model.generate_content(prompt)
            if not response.text:
                raise RuntimeError("Empty response from Gemini API")
            return response.text

        except genai.APIError as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")