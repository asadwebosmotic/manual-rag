from typing import List, Dict

class SlidingWindowSession:
    def __init__(self, max_turns: int = 20):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """
        Add a user-assistant turn to the session history.
        
        Args:
            user_text: User input text.
            assistant_text: Assistant response text.
        """
        self.history.append({"text": user_text, "role": "USER"})
        self.history.append({"text": assistant_text, "role": "ASSISTANT"})
        # Keep only the last max_turns pairs (user + assistant)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_window_history(self) -> List[Dict[str, str]]:
        """
        Get the full session history.
        
        Returns:
            List of history turns.
        """
        return self.history

    def get_messages(self, role_mapping: Dict[str, str] = None) -> List[Dict[str, str]]:
        """
        Get history in a format suitable for LLMs, with customizable role mapping.
        
        Args:
            role_mapping: Optional mapping of internal roles to LLM-specific roles.
                         E.g., {"USER": "user", "ASSISTANT": "assistant"} for Gemini.
        
        Returns:
            List of messages in the format [{"role": str, "content": str}, ...].
        """
        role_mapping = role_mapping or {"USER": "USER", "ASSISTANT": "ASSISTANT"}
        return [
            {"role": role_mapping[turn["role"]], "content": turn["text"]}
            for turn in self.history
        ]