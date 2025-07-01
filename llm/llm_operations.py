'''importing necessary libraries'''
from config import settings
import google.generativeai as genai

class GeminiChat:
    '''constructor for initializing the gemini model and chat history'''
    def __init__(self):
        genai.configure(api_key = settings.GEMINI_API_KEY) #configure the gemini api key
        self.model = genai.GenerativeModel("gemini-2.5-flash") #initialize the gemini model
        self.chat = self.model.start_chat(history = []) #initialize the chat history

    '''Method to generate a response based on user input (and optional context)'''
    def generate_response(self, user_input, context = None):
        if context: # If some context is provided (like retrieved docs or past info), attach it to the prompt.
            # Merge context and user input into a single prompt for better-informed responses.
            prompt = f"Context: {context}\n\nUser: {user_input}"
        else: # If no context, just use the user's input.
            prompt = user_input

        # send the prompt to the gemini and get the response
        response = self.chat.send_message(prompt)

        # Return only the text content of the response (stripping metadata etc.)
        return response.text
    
    # Method to get the chat history
    def get_chat_history(self):
        # Returns a list of all previous messages in the conversation (user + AI)
        return self.chat.history