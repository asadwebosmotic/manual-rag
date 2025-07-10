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

        # Updated prompt with stronger instructions + few-shot example
        self.prompt_template = """You are a helpful, smart assistant who uses the given document context and chat history to answer user questions.

        NEVER ignore the provided context. If no context is available, respond using your general knowledge — but indicate that you're not using document information.

        Here’s how to behave:

        Example:
        User: Who is Batman?
        Assistant: Batman is a superhero from DC Comics...
        User: Who are his allies?
        Assistant: Batman’s allies include Robin, Alfred Pennyworth, Batgirl...

        ---

        Chat so far:
        {history}

        Relevant document excerpts:
        {context}

        Current User Question:
        {user_input}

        Now respond as the assistant:"""

    def generate_response(
        self,
        user_input: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generates a response from Gemini using document context and chat memory.
        """

        try:
            if not user_input.strip():
                raise ValueError("User input cannot be empty")

            # Format chat history as a readable back-and-forth
            history_str = ""
            if chat_history:
                for turn in chat_history:
                    role = "User" if turn["role"].upper() == "USER" else "Assistant"
                    history_str += f"{role}: {turn['content'].strip()}\n"
            else:
                history_str = "No prior conversation."

            # Build the full prompt
            prompt = self.prompt_template.format(
                context=context.strip() if context else "No relevant context retrieved.",
                history=history_str.strip(),
                user_input=user_input.strip()
            )

            print("🧠 FINAL PROMPT SENT TO GEMINI:\n", prompt[:1000])  # Preview first 1000 chars

            # Generate response from Gemini
            response = self.model.generate_content(prompt)

            if not response.text:
                raise RuntimeError("Empty response from Gemini API")

            return response.text.strip()

        except genai.APIError as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")
