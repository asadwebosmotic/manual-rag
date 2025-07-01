'''craeting SlidingWindowSession class to incapsulate the logic'''
class SlidingWindowSession:
    def __init__(self, max_turns: int = 10):
        self.history = []
        self.max_turns = max_turns

    def add_turn(self, text, user, assistant):
        self.history.append({"text": text, "role": user}, {"text": text, "role": assistant})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]  # Keep only last N

    def get_window_history(self):
        return self.history
